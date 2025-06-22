#!/usr/bin/env python
# Copyright 2014-2020 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
MP2-F12 (In testing)

Refs:
* JCC 32, 2492 (2011); DOI:10.1002/jcc.21825
* JCP 139, 084112 (2013); DOI:10.1063/1.4818753

Follows the first reference more closely.

With strong orthogonalization ansatz 2
'''

import warnings
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto, scf
from pyscf import ao2mo
from pyscf.scf import jk
from pyscf.mp import mp2, gmp2

warnings.warn('Module MP2-F12 is under testing')

def make_ghost_atoms(auxmol):
    # auxmol should provide auxilliary basis functions but not extra nuclei.
    # WARNING: this is a bit hacky. Check.
    for atm in auxmol._atm:
        atm[0] = 0  # remove charge
    atoms = []
    for atom in auxmol._atom:
        atoms.append(("GHOST-"+atom[0], atom[1]))
    auxmol._atom = atoms
    basis = {}
    for atm, bas in auxmol._basis.items():
        basis["GHOST-"+atm] = bas
    auxmol._basis = basis
    return auxmol

# The cabs space, the complimentary space to the OBS.
def find_cabs(mol, auxmol, lindep=1e-8):
    cabs_mol = gto.conc_mol(mol, auxmol)
    nao = mol.nao_nr()
    s = cabs_mol.intor_symmetric('int1e_ovlp')
    ls12 = scipy.linalg.solve(s[:nao,:nao], s[:nao,nao:], assume_a='pos')
    s[nao:,nao:] -= s[nao:,:nao].dot(ls12)
    w, v = scipy.linalg.eigh(s[nao:,nao:])
    c2 = v[:,w>lindep]/numpy.sqrt(w[w>lindep])
    c1 = ls12.dot(c2)
    cabs_coeff_spatial = numpy.vstack((-c1,c2))
    cabs_coeff = numpy.zeros((2*cabs_coeff_spatial.shape[0], 2*cabs_coeff_spatial.shape[1]))
    for icol in range(cabs_coeff_spatial.shape[1]):
        cabs_coeff[cabs_coeff_spatial.shape[0]:,2*icol] = cabs_coeff_spatial[:,icol]
        cabs_coeff[:cabs_coeff_spatial.shape[0],2*icol+1] = cabs_coeff_spatial[:,icol]
    return cabs_mol, cabs_coeff

def trans(eri, mos):
    naoi, nmoi = mos[0].shape
    naoj, nmoj = mos[1].shape
    naok, nmok = mos[2].shape
    naol, nmol = mos[3].shape
    eri1 = numpy.dot(mos[0].T.conj(), eri.reshape(naoi,-1))
    eri1 = eri1.reshape(nmoi,naoj,naok,naol)

    eri1 = numpy.dot(mos[1].T, eri1.transpose(1,0,2,3).reshape(naoj,-1))
    eri1 = eri1.reshape(nmoj,nmoi,naok,naol).transpose(1,0,2,3)

    eri1 = numpy.dot(eri1.transpose(0,1,3,2).reshape(-1,naok), mos[2].conj())
    eri1 = eri1.reshape(nmoi,nmoj,naol,nmok).transpose(0,1,3,2)

    eri1 = numpy.dot(eri1.reshape(-1,naol), mos[3])
    eri1 = eri1.reshape(nmoi,nmoj,nmok,nmol)
    return eri1


def trans_spinorb(eri, mos, apply_sxy=False, apply_both_sxy_svw=False):
    if apply_sxy:
        f_abab = 3.0/8.0
        f_baab = 1.0/8.0
    elif apply_both_sxy_svw:
        # 3/8 * 3/8 + 1/8 * 1/8 = 10/64 and 3/8*1/8 + 1/8*3/8 = 6/64.
        f_abab = 10.0/64.0
        f_baab = 6.0/64.0
    else:
        f_abab = 1.0
        f_baab = 0.0

    nao_halves = [mo.shape[0]//2 for mo in mos]
    # aaaa
    eri_samesp = trans(eri, (mos[0][:nao_halves[0]], mos[1][:nao_halves[1]], mos[2][:nao_halves[2]], mos[3][:nao_halves[3]]))
    # bbbb
    eri_samesp += trans(eri, (mos[0][nao_halves[0]:], mos[1][nao_halves[1]:], mos[2][nao_halves[2]:], mos[3][nao_halves[3]:]))
    # aabb
    eri_aabb = f_abab*trans(eri, (mos[0][:nao_halves[0]], mos[1][:nao_halves[1]], mos[2][nao_halves[2]:], mos[3][nao_halves[3]:]))
    # baab  # usually zero but needed for the Sxy operator where spatial orb but not spin are swapped in second term.
    eri_aabb += f_baab*trans(eri, (mos[0][nao_halves[0]:], mos[1][:nao_halves[1]], mos[2][:nao_halves[2]], mos[3][nao_halves[3]:])).transpose(2,1,0,3)
    del eri
    # Convert to physics notation, antisymmetrize and apply Sxy and Svw operators.
    eri_samesp = eri_samesp.transpose(0,2,1,3) - eri_samesp.transpose(0,2,3,1)
    eri_aabb = eri_aabb.transpose(0,2,1,3) - eri_aabb.transpose(0,2,3,1)

    if apply_sxy or apply_both_sxy_svw:
        eri_samesp = (3.0/8.0)*eri_samesp + (1.0/8.0)*eri_samesp.transpose(1,0,2,3)
        if apply_both_sxy_svw:  
            eri_samesp = (3.0/8.0)*eri_samesp + (1.0/8.0)*eri_samesp.transpose(0,1,3,2)

    # WARNING: i < j and similarly for dummy indices in einsums. Therefore make sure that
    # arrays are aligned such that this i < j here also restricts dummy indices in einsums.
    # i < j
    return numpy.triu(eri_samesp + eri_aabb + eri_aabb.transpose(1,0,3,2))

def amplitudes_to_vector(t1, out=None):
    nocc, naux = t1.shape
    nov = nocc * naux
    vector = t1.ravel()
    return vector

def vector_to_amplitudes(vector, nocc):
    naux = len(vector)//nocc
    t1 = vector.copy().reshape((nocc,naux))
    return t1

def energy_f12(mf, mp, auxmol, zeta, frozen=0):
    logger.info(mf, '******** MP2-F12 (In testing) ********')
    auxmol = make_ghost_atoms(auxmol)
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    nocc = numpy.count_nonzero(mf.mo_occ == 1)
    if getattr(mo_coeff, 'orbspin', None) is not None:
        print(mo_coeff.orbspin)
        print("orbspin not supported yet. Spin is estimated by which AOs are nonzero.")

    cabs_mol, cabs_coeff = find_cabs(mol, auxmol)
    nao, nmo = mo_coeff.shape
    nca = cabs_coeff.shape[0]
    mo_o = mo_coeff[:,:nocc]
    Pcoeff = numpy.vstack((mo_coeff[:nao//2], numpy.zeros(((nca-nao)//2, nmo)),mo_coeff[nao//2:], numpy.zeros(((nca-nao)//2, nmo))))
    Pcoeff = numpy.hstack((Pcoeff, cabs_coeff))
    obs = (0, mol.nbas)
    cbs = (0, cabs_mol.nbas)
    mol.set_f12_zeta(zeta)
    Y = -mol.intor('int2e_yp', shls_slice=obs+obs+obs+obs)/zeta
    Y = trans_spinorb(Y, [mo_o]*4, apply_sxy=True)
    print("done Y")

    cabs_mol.set_f12_zeta(zeta)
    RPQRS = -cabs_mol.intor('int2e_stg', shls_slice=cbs+cbs+cbs+cbs)/zeta
    print("done R intor")
    RPQRS = trans_spinorb(RPQRS, [Pcoeff, Pcoeff, Pcoeff, Pcoeff], apply_sxy=True)
    print("done R")

    cabs_mol.set_f12_zeta(zeta*2)
    RbarmnRS = cabs_mol.intor('int2e_stg', shls_slice=obs+cbs+obs+cbs)/(zeta**2)
    RbarmnRS = trans_spinorb(RbarmnRS, [mo_o, Pcoeff, mo_o, Pcoeff], apply_both_sxy_svw=True)
    print("done Rbar")

    tau = RbarmnRS[:nocc,:nocc,:nocc,:nocc].copy() * zeta**2

    vPQij = cabs_mol.intor('int2e', shls_slice=cbs+obs+cbs+obs)
    vPQij = trans_spinorb(vPQij, [Pcoeff,mo_o,Pcoeff,mo_o])
    print("done v")

    # v is in chemist's notation here and is in spatial orb form. dm is the spinorbital dm.
    fPQ = mf.get_hcore(cabs_mol)
    tmpk = fPQ.copy()
    dm = numpy.dot(mo_o, mo_o.T)
    v = cabs_mol.intor('int2e', shls_slice=cbs+cbs+obs+obs)
    v = v.reshape(nca//2,nca//2,nao//2,nao//2)
    # ij aa PQ aa
    fPQ[:nca//2,:nca//2] += numpy.einsum('pqij,ji->pq', v, dm[:nao//2,:nao//2])
    # ij aa PQ bb
    fPQ[nca//2:,nca//2:] += numpy.einsum('pqij,ji->pq', v, dm[:nao//2,:nao//2])
    # ij bb PQ aa
    fPQ[:nca//2,:nca//2] += numpy.einsum('pqij,ji->pq', v, dm[nao//2:,nao//2:])
    # ij bb PQ bb
    fPQ[nca//2:,nca//2:] += numpy.einsum('pqij,ji->pq', v, dm[nao//2:,nao//2:])
    fPQ = reduce(numpy.dot, (Pcoeff.T, fPQ, Pcoeff))
    v = cabs_mol.intor('int2e', shls_slice=cbs+obs+obs+cbs)
    v = v.reshape(nca//2,nao//2,nao//2,nca//2)
    kPQ = numpy.zeros((nca, nca))
    # ij aa PQ aa
    kPQ[:nca//2,:nca//2] += numpy.einsum('pijq,ij->pq', v, dm[:nao//2,:nao//2])
    # ij bb PQ bb
    kPQ[nca//2:,nca//2:] += numpy.einsum('pijq,ij->pq', v, dm[nao//2:,nao//2:])
    kPQ = reduce(numpy.dot, (Pcoeff.T, kPQ, Pcoeff))
    hPQ = fPQ.copy()
    fPQ = hPQ.copy() - kPQ.copy()

    f1PQ = numpy.zeros(fPQ.shape)
    f1PQ[:nocc,nocc:] = fPQ[:nocc,nocc:]
    f1PQ[nocc:,:nocc] = fPQ[nocc:,:nocc]
    f0PQ = fPQ.copy() - f1PQ.copy()
    
    V = Y.copy()
    V-= numpy.einsum('mnpq,pqij->mnij', RPQRS[:nocc,:nocc,:nmo,:nmo], vPQij[:nmo,:nmo,:nocc,:nocc])
    V-= numpy.einsum('mnlc,lcij->mnij', RPQRS[:nocc,:nocc,:nocc,nmo:], vPQij[:nocc,nmo:,:nocc,:nocc])
    V-= numpy.einsum('mncl,clij->mnij', RPQRS[:nocc,:nocc,nmo:,:nocc], vPQij[nmo:,:nocc,:nocc,:nocc])
    emp2_f12 = numpy.einsum('ijij->', V[frozen:,frozen:,frozen:,frozen:])*2
    
    X_mniP = RbarmnRS[:nocc,:nocc,:nocc,:].copy()
    X_mniP-= numpy.einsum('mnpq,pqiP->mniP', RPQRS[:nocc,:nocc,:nmo,:nmo], RPQRS[:nmo,:nmo,:nocc,:])
    X_mniP-= numpy.einsum('mnlc,lciP->mniP', RPQRS[:nocc,:nocc,:nocc,nmo:], RPQRS[:nocc,nmo:,:nocc,:])
    X_mniP-= numpy.einsum('mncl,cliP->mniP', RPQRS[:nocc,:nocc,nmo:,:nocc], RPQRS[nmo:,:nocc,:nocc,:])
    
    X_mnPj = RbarmnRS[:nocc,:nocc,:,:nocc].copy()
    X_mnPj-= numpy.einsum('mnpq,pqPj->mnPj', RPQRS[:nocc,:nocc,:nmo,:nmo], RPQRS[:nmo,:nmo,:,:nocc])
    X_mnPj-= numpy.einsum('mnlc,lcPj->mnPj', RPQRS[:nocc,:nocc,:nocc,nmo:], RPQRS[:nocc,nmo:,:,:nocc])
    X_mnPj-= numpy.einsum('mncl,clPj->mnPj', RPQRS[:nocc,:nocc,nmo:,:nocc], RPQRS[nmo:,:nocc,:,:nocc])

    t_PQij = numpy.einsum('RQij,RP->PQij', RPQRS[:,:,:nocc,:nocc], kPQ + fPQ)
    t_PQij+= numpy.einsum('PRij,RQ->PQij', RPQRS[:,:,:nocc,:nocc], kPQ + fPQ)
    t_PQij-= numpy.einsum('PQRj,Ri->PQij', RPQRS[:,:,:,:nocc], kPQ[:,:nocc] + fPQ[:,:nocc])
    t_PQij-= numpy.einsum('PQiR,Rj->PQij', RPQRS[:,:,:nocc,:], kPQ[:,:nocc] + fPQ[:,:nocc])
    
    T = tau.copy()
    T-= numpy.einsum('mnpq,pqij->mnij', RPQRS[:nocc,:nocc,:nmo,:nmo], t_PQij[:nmo,:nmo])
    T-= numpy.einsum('mncl,clij->mnij', RPQRS[:nocc,:nocc,nmo:,:nocc], t_PQij[nmo:,:nocc])
    T-= numpy.einsum('mnlc,lcij->mnij', RPQRS[:nocc,:nocc,:nocc,nmo:], t_PQij[:nocc,nmo:])

    p_PQij = numpy.einsum('RQij,RP->PQij', RPQRS[:,:,:nocc,:nocc], kPQ + f1PQ)
    p_PQij += numpy.einsum('PRij,RQ->PQij', RPQRS[:,:,:nocc,:nocc], kPQ + f1PQ)

    P = numpy.einsum('mncd,cdij->mnij', RPQRS[:nocc,:nocc,nmo:,nmo:], p_PQij[nmo:,nmo:])
    P+= numpy.einsum('mnca,caij->mnij', RPQRS[:nocc,:nocc,nmo:,nocc:nmo], p_PQij[nmo:,nocc:nmo])
    P+= numpy.einsum('mnac,acij->mnij', RPQRS[:nocc,:nocc,nocc:nmo,nmo:], p_PQij[nocc:nmo,nmo:])

    q_PQij = numpy.einsum('PQRj,Ri->PQij', RPQRS[:,:,:,:nocc], kPQ[:,:nocc] + f1PQ[:,:nocc])
    q_PQij+= numpy.einsum('PQiR,Rj->PQij', RPQRS[:,:,:nocc,:], kPQ[:,:nocc] + f1PQ[:,:nocc])

    Q = numpy.einsum('mnPj,Pi->mnij',RbarmnRS[:nocc,:nocc,:,:nocc],kPQ[:,:nocc] + f1PQ[:,:nocc])
    Q+= numpy.einsum('mniP,Pj->mnij',RbarmnRS[:nocc,:nocc,:nocc,:],kPQ[:,:nocc] + f1PQ[:,:nocc])
    Q-= numpy.einsum('mnpq,pqij->mnij', RPQRS[:nocc,:nocc,:nmo,:nmo], q_PQij[:nmo,:nmo])
    Q-= numpy.einsum('mncl,clij->mnij', RPQRS[:nocc,:nocc,nmo:,:nocc], q_PQij[nmo:,:nocc])
    Q-= numpy.einsum('mnlc,lcij->mnij', RPQRS[:nocc,:nocc,:nocc,nmo:], q_PQij[:nocc,nmo:])

    C = numpy.einsum('mncb,ac->mnab',RPQRS[:nocc,:nocc,nmo:,nocc:nmo], f0PQ[nocc:nmo,nmo:])
    C+= numpy.einsum('mnac,bc->mnab',RPQRS[:nocc,:nocc,nocc:nmo,nmo:], f0PQ[nocc:nmo,nmo:])

    E = numpy.einsum('mnab,abij->mnij',C,RPQRS[nocc:nmo,nocc:nmo,:nocc,:nocc])

    B = numpy.einsum('mniP,Pj->mnij', X_mniP, f0PQ[:,:nocc])
    B += numpy.einsum('mnPj,Pi->mnij', X_mnPj, f0PQ[:,:nocc])
    B += T
    B -= P
    B += Q
    B -= E
    B = 0.5*(B + B.transpose(2,3,0,1))
    emp2_f12 += numpy.einsum('ijij->', B[frozen:,frozen:,frozen:,frozen:])
    emp2_f12-= numpy.einsum('ijkj,ki', X_mniP[frozen:,frozen:,frozen:,frozen:nocc], fPQ[frozen:nocc,frozen:nocc])
    emp2_f12-= numpy.einsum('ijil,lj', X_mniP[frozen:,frozen:,frozen:,frozen:nocc], fPQ[frozen:nocc,frozen:nocc])
    print("f12 extra", emp2_f12)

    #CABS
    inv_e_ai = 1/(mo_energy[nocc:nmo,None] - mo_energy[None,:nocc])
    Pmo_energy = fPQ.diagonal()
    inv_e_ic = 1/(mo_energy[:nocc,None] - Pmo_energy[None,nmo:])
    f_wig = fPQ[:nocc,nmo:] - numpy.einsum("ac,ai,ia->ic", fPQ[nocc:nmo,nmo:], inv_e_ai, fPQ[:nocc, nocc:nmo])
    t_s = numpy.einsum("ic,ic->ic", f_wig, inv_e_ic)
    cabs_energy = numpy.einsum("ci,ic->", f_wig.T.conj(), t_s)
    cabs_energy_new = numpy.inf
    print("init CABS energy", cabs_energy)
    adiis = lib.diis.DIIS()
    icycle = 0
    diff_t_norm = 1.0
    while (abs(cabs_energy - cabs_energy_new) > 1e-10 or diff_t_norm > 1e-10) and icycle < 300:
        icycle += 1
        cabs_energy = cabs_energy_new
        t_s_new = numpy.einsum("dc,id->ic", fPQ[nmo:,nmo:], t_s)
        t_s_new -= numpy.einsum("cc,ic->ic", fPQ[nmo:,nmo:], t_s)
        t_s_new -= numpy.einsum("ij,jc->ic", fPQ[:nocc,:nocc], t_s)
        t_s_new += numpy.einsum("ii,ic->ic", fPQ[:nocc,:nocc], t_s)
        t_s_new -= numpy.einsum("ac,ai,da,id->ic", fPQ[nocc:nmo,nmo:], inv_e_ai, fPQ[nmo:, nocc:nmo], t_s)
        t_s_new += f_wig
        t_s_new = numpy.einsum("ic,ic->ic", t_s_new, inv_e_ic)
        vec = amplitudes_to_vector(t_s_new)
        t_s_new = vector_to_amplitudes(adiis.update(vec), nocc)
        diff_t_norm = numpy.linalg.norm(t_s_new - t_s)
        t_s = t_s_new
        cabs_energy_new = numpy.einsum("ci,ic->", f_wig.T.conj(), t_s_new)
        print("CABS cycle", icycle, cabs_energy_new, abs(cabs_energy - cabs_energy_new), diff_t_norm)
    # Check whether R in eq. 35 (Turbomole paper) is really zero.
    tmp = f_wig + numpy.einsum("dc,id->ic", fPQ[nmo:,nmo:], t_s)
    tmp -= numpy.einsum("ij,jc->ic", fPQ[:nocc,:nocc], t_s)
    tmp -= numpy.einsum("ac,ai,da,id->ic", fPQ[nocc:nmo,nmo:], inv_e_ai, fPQ[nmo:, nocc:nmo], t_s)
    if numpy.max(numpy.abs(tmp)) < 1e-8:
        print("CABS SUCCESS. max R", numpy.max(numpy.abs(tmp)))
        print("CABS energy", cabs_energy_new)
    else:
        print("CABS FAIL. max R", numpy.max(numpy.abs(tmp)))
    print("return f12 extra (without CABS)", emp2_f12)
    return emp2_f12


if __name__ == '__main__':
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = 'C 0 0 0; C 0 0 2'
    mol.basis = "ccpvdz"
    mol.build()
    mf = scf.RHF(mol)
    mf.max_cycle = 2
    mf.kernel()

    mf = mf.to_ghf()

    mymp = gmp2.GMP2(mf)
    e = mymp.kernel()[0]

    auxmol = mol.copy()
    auxmol.basis = "ccpvtz"
    #auxmol.basis = ('ccpvqz-fit', 'cc-pVQZ-F12-OptRI')
    #auxmol.basis = 'cc-pVQZ'
    auxmol.build(False, False)
    print('MP2', e)
    e+= energy_f12(mf, auxmol, 1.)
    print('MP2-F12', e)
    print('e_tot', e+mf.e_tot)
