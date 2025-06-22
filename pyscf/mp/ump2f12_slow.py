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
from pyscf.mp import mp2, ump2

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
    cabs_coeff = numpy.array([cabs_coeff_spatial, cabs_coeff_spatial])
    return cabs_mol, cabs_coeff

def amplitudes_to_vector(t1, out=None):
    nocc, naux = t1.shape
    nov = nocc * naux
    vector = t1.ravel()
    return vector

def vector_to_amplitudes(vector, nocc):
    naux = len(vector)//nocc
    t1 = vector.copy().reshape((nocc,naux))
    return t1

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


def trans_unres(eri, mos, apply_sxy=False, apply_both_sxy_svw=False):
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

    # aaaa
    eri_aa = trans(eri, (mos[0][0], mos[1][0], mos[2][0], mos[3][0]))
    # bbbb
    eri_bb = trans(eri, (mos[0][1], mos[1][1], mos[2][1], mos[3][1]))
    # aabb
    eri_ab = f_abab*trans(eri, (mos[0][0], mos[1][0], mos[2][1], mos[3][1]))
    # baab  # usually zero but needed for the Sxy operator where spatial orb but not spin are swapped in second term.
    eri_ab += f_baab*trans(eri, (mos[0][1], mos[1][0], mos[2][0], mos[3][1])).transpose(2,1,0,3)
    del eri
    
    # Convert to physics notation, antisymmetrize and apply Sxy and Svw operators.
    eri_aa = eri_aa.transpose(0,2,1,3) - eri_aa.transpose(0,2,3,1)
    eri_bb = eri_bb.transpose(0,2,1,3) - eri_bb.transpose(0,2,3,1)
    eri_ab = eri_ab.transpose(0,2,1,3)

    if apply_sxy or apply_both_sxy_svw:
        eri_aa = (3.0/8.0)*eri_aa + (1.0/8.0)*eri_aa.transpose(1,0,2,3)
        eri_bb = (3.0/8.0)*eri_bb + (1.0/8.0)*eri_bb.transpose(1,0,2,3)
        if apply_both_sxy_svw:  
            eri_aa = (3.0/8.0)*eri_aa + (1.0/8.0)*eri_aa.transpose(1,0,2,3)
            eri_bb = (3.0/8.0)*eri_bb + (1.0/8.0)*eri_bb.transpose(1,0,2,3)

    # WARNING: i < j and similarly for dummy indices in einsums. Therefore make sure that
    # arrays are aligned such that this i < j here also restricts dummy indices in einsums.
    # i < j
    return numpy.triu(eri_aa), numpy.triu(eri_bb), eri_ab


def energy_f12(mf, mp, auxmol, zeta, frozen=0):
    logger.info(mf, '******** MP2-F12 (In testing) ********')
    auxmol = make_ghost_atoms(auxmol)
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    mo_energy = mf.mo_energy
    nocc_a = numpy.count_nonzero(mf.mo_occ[0] == 1)
    nocc_b = numpy.count_nonzero(mf.mo_occ[1] == 1)

    cabs_mol, cabs_coeff = find_cabs(mol, auxmol)
    nao, nmo = mo_coeff[0].shape
    nca = cabs_coeff[0].shape[0]
    mo_o_a = mo_coeff[0,:,:nocc_a]
    mo_o_b = mo_coeff[1,:,:nocc_b]
    mo_o = [mo_o_a, mo_o_b]
    Pcoeff_a = numpy.vstack((mo_coeff[0], numpy.zeros((nca-nao, nmo))))
    Pcoeff_a = numpy.hstack((Pcoeff_a, cabs_coeff[0]))
    Pcoeff_b = numpy.vstack((mo_coeff[1], numpy.zeros((nca-nao, nmo))))
    Pcoeff_b = numpy.hstack((Pcoeff_b, cabs_coeff[1]))
    Pcoeff = numpy.array([Pcoeff_a, Pcoeff_b])
    obs = (0, mol.nbas)
    cbs = (0, cabs_mol.nbas)
    mol.set_f12_zeta(zeta)
    Y = -mol.intor('int2e_yp', shls_slice=obs+obs+obs+obs)/zeta
    Y_aa, Y_bb, Y_ab = trans_unres(Y, [mo_o]*4, apply_sxy=True)
    del Y
    print("done Y")

    cabs_mol.set_f12_zeta(zeta)
    RPQRS = -cabs_mol.intor('int2e_stg', shls_slice=cbs+cbs+cbs+cbs)/zeta
    print("done R intor")
    RPQRS_aa, RPQRS_bb, RPQRS_ab = trans_unres(RPQRS, [Pcoeff, Pcoeff, Pcoeff, Pcoeff], apply_sxy=True)
    del RPQRS
    print("done R")

    cabs_mol.set_f12_zeta(zeta*2)
    RbarmnRS = cabs_mol.intor('int2e_stg', shls_slice=obs+cbs+obs+cbs)/(zeta**2)
    RbarmnRS_aa, RbarmnRS_bb, RbarmnRS_ab = trans_unres(RbarmnRS, [mo_o, Pcoeff, mo_o, Pcoeff], apply_both_sxy_svw=True)
    del RbarmnRS
    print("done Rbar")

    tau_aa = RbarmnRS_aa[:nocc_a,:nocc_a,:nocc_a,:nocc_a].copy() * zeta**2
    tau_bb = RbarmnRS_bb[:nocc_b,:nocc_b,:nocc_b,:nocc_b].copy() * zeta**2
    tau_ab = RbarmnRS_ab[:nocc_a,:nocc_b,:nocc_a,:nocc_b].copy() * zeta**2

    vPQij = cabs_mol.intor('int2e', shls_slice=cbs+obs+cbs+obs)
    vPQij_aa, vPQij_bb, vPQij_ab = trans_unres(vPQij, [Pcoeff,mo_o,Pcoeff,mo_o])
    del vPQij
    print("done v")

    # v is in chemist's notation here and is in spatial orb form. dm is the unrestricted spatial dm.
    fPQ = mf.get_hcore(cabs_mol)
    fPQ_aa = fPQ.copy()
    fPQ_bb = fPQ.copy()
    del fPQ
    dm_aa = numpy.dot(mo_o_a, mo_o_a.T)
    dm_bb = numpy.dot(mo_o_b, mo_o_b.T)
    v = cabs_mol.intor('int2e', shls_slice=cbs+cbs+obs+obs)
    v = v.reshape(nca,nca,nao,nao)
    # ij aa PQ aa
    fPQ_aa += numpy.einsum('pqij,ji->pq', v, dm_aa)
    # ij aa PQ bb
    fPQ_bb += numpy.einsum('pqij,ji->pq', v, dm_aa)
    # ij bb PQ aa
    fPQ_aa += numpy.einsum('pqij,ji->pq', v, dm_bb)
    # ij bb PQ bb
    fPQ_bb += numpy.einsum('pqij,ji->pq', v, dm_bb)
    fPQ_aa = reduce(numpy.dot, (Pcoeff_a.T, fPQ_aa, Pcoeff_a))
    fPQ_bb = reduce(numpy.dot, (Pcoeff_b.T, fPQ_bb, Pcoeff_b))
    v = cabs_mol.intor('int2e', shls_slice=cbs+obs+obs+cbs)
    v = v.reshape(nca,nao,nao,nca)
    kPQ_aa = numpy.zeros((nca, nca))
    kPQ_bb = numpy.zeros((nca, nca))
    # ij aa PQ aa
    kPQ_aa += numpy.einsum('pijq,ij->pq', v, dm_aa)
    # ij bb PQ bb
    kPQ_bb += numpy.einsum('pijq,ij->pq', v, dm_bb)
    kPQ_aa = reduce(numpy.dot, (Pcoeff_a.T, kPQ_aa, Pcoeff_a))
    kPQ_bb = reduce(numpy.dot, (Pcoeff_b.T, kPQ_bb, Pcoeff_b))
    hPQ_aa = fPQ_aa.copy()
    hPQ_bb = fPQ_bb.copy()
    fPQ_aa = hPQ_aa.copy() - kPQ_aa.copy()
    fPQ_bb = hPQ_bb.copy() - kPQ_bb.copy()

    f1PQ_aa = numpy.zeros(fPQ_aa.shape)
    f1PQ_bb = numpy.zeros(fPQ_bb.shape)
    f1PQ_aa[:nocc_a,nocc_a:] = fPQ_aa[:nocc_a,nocc_a:]
    f1PQ_bb[:nocc_b,nocc_b:] = fPQ_bb[:nocc_b,nocc_b:]
    f1PQ_aa[nocc_a:,:nocc_a] = fPQ_aa[nocc_a:,:nocc_a]
    f1PQ_bb[nocc_b:,:nocc_b] = fPQ_bb[nocc_b:,:nocc_b]
    f0PQ_aa = fPQ_aa.copy() - f1PQ_aa.copy()
    f0PQ_bb = fPQ_bb.copy() - f1PQ_bb.copy()

    def V_contribution(Y, RPQRS, vPQij, nocc_m, nocc_n):
        V = Y.copy()
        V-= numpy.einsum('mnpq,pqij->mnij', RPQRS[:nocc_m,:nocc_n,:nmo,:nmo], vPQij[:nmo,:nmo,:nocc_m,:nocc_n])
        V-= numpy.einsum('mnlc,lcij->mnij', RPQRS[:nocc_m,:nocc_n,:nocc_m,nmo:], vPQij[:nocc_m,nmo:,:nocc_m,:nocc_n])
        V-= numpy.einsum('mncl,clij->mnij', RPQRS[:nocc_m,:nocc_n,nmo:,:nocc_n], vPQij[nmo:,:nocc_n,:nocc_m,:nocc_n])
        e = numpy.einsum('ijij->', V[frozen:,frozen:,frozen:,frozen:])*2
        return e

    emp2_f12 = V_contribution(Y_aa, RPQRS_aa, vPQij_aa, nocc_a, nocc_a)
    emp2_f12 += V_contribution(Y_bb, RPQRS_bb, vPQij_bb, nocc_b, nocc_b)
    emp2_f12 += V_contribution(Y_ab, RPQRS_ab, vPQij_ab, nocc_a, nocc_b)
    
    def get_X_mniP(RbarmnRS, RPQRS, nocc_m, nocc_n):
        X_mniP = RbarmnRS[:nocc_m,:nocc_n,:nocc_m,:].copy()
        X_mniP-= numpy.einsum('mnpq,pqiP->mniP', RPQRS[:nocc_m,:nocc_n,:nmo,:nmo], RPQRS[:nmo,:nmo,:nocc_m,:])
        X_mniP-= numpy.einsum('mnlc,lciP->mniP', RPQRS[:nocc_m,:nocc_n,:nocc_m,nmo:], RPQRS[:nocc_m,nmo:,:nocc_m,:])
        X_mniP-= numpy.einsum('mncl,cliP->mniP', RPQRS[:nocc_m,:nocc_n,nmo:,:nocc_n], RPQRS[nmo:,:nocc_n,:nocc_m,:])
        return X_mniP

    X_mniP_aa = get_X_mniP(RbarmnRS_aa, RPQRS_aa, nocc_a, nocc_a)
    X_mniP_bb = get_X_mniP(RbarmnRS_bb, RPQRS_bb, nocc_b, nocc_b)
    X_mniP_ab = get_X_mniP(RbarmnRS_ab, RPQRS_ab, nocc_a, nocc_b)

    def X_contribution(X_mniP, fPQ_i, fPQ_j, nocc_i, nocc_j):
        e = -numpy.einsum('ijkj,ki', X_mniP[frozen:,frozen:,frozen:,frozen:nocc_j], fPQ_i[frozen:nocc_i,frozen:nocc_i])
        e -= numpy.einsum('ijil,lj', X_mniP[frozen:,frozen:,frozen:,frozen:nocc_j], fPQ_j[frozen:nocc_j,frozen:nocc_j])
        return e

    emp2_f12 += X_contribution(X_mniP_aa, fPQ_aa, fPQ_aa, nocc_a, nocc_a)
    emp2_f12 += X_contribution(X_mniP_bb, fPQ_bb, fPQ_bb, nocc_b, nocc_b)
    emp2_f12 += X_contribution(X_mniP_ab, fPQ_aa, fPQ_bb, nocc_a, nocc_b)

    def get_X_mnPj(RbarmnRS, RPQRS, nocc_m, nocc_n):
        X_mnPj = RbarmnRS[:nocc_m,:nocc_n,:,:nocc_n].copy()
        X_mnPj-= numpy.einsum('mnpq,pqPj->mnPj', RPQRS[:nocc_m,:nocc_n,:nmo,:nmo], RPQRS[:nmo,:nmo,:,:nocc_n])
        X_mnPj-= numpy.einsum('mnlc,lcPj->mnPj', RPQRS[:nocc_m,:nocc_n,:nocc_m,nmo:], RPQRS[:nocc_m,nmo:,:,:nocc_n])
        X_mnPj-= numpy.einsum('mncl,clPj->mnPj', RPQRS[:nocc_m,:nocc_n,nmo:,:nocc_n], RPQRS[nmo:,:nocc_n,:,:nocc_n])
        return X_mnPj

    X_mnPj_aa = get_X_mnPj(RbarmnRS_aa, RPQRS_aa, nocc_a, nocc_a)
    X_mnPj_bb = get_X_mnPj(RbarmnRS_bb, RPQRS_bb, nocc_b, nocc_b)
    X_mnPj_ab = get_X_mnPj(RbarmnRS_ab, RPQRS_ab, nocc_a, nocc_b)

    def get_t_PQij(RPQRS, kPQ_fPQ_i, kPQ_fPQ_j, nocc_i, nocc_j):
        t_PQij = numpy.einsum('RQij,RP->PQij', RPQRS[:,:,:nocc_i,:nocc_j], kPQ_fPQ_i)
        t_PQij+= numpy.einsum('PRij,RQ->PQij', RPQRS[:,:,:nocc_i,:nocc_j], kPQ_fPQ_j)
        t_PQij-= numpy.einsum('PQRj,Ri->PQij', RPQRS[:,:,:,:nocc_j], kPQ_fPQ_i[:,:nocc_i])
        t_PQij-= numpy.einsum('PQiR,Rj->PQij', RPQRS[:,:,:nocc_i,:], kPQ_fPQ_j[:,:nocc_j])
        return t_PQij

    t_PQij_aa = get_t_PQij(RPQRS_aa, kPQ_aa + fPQ_aa, kPQ_aa + fPQ_aa, nocc_a, nocc_a)
    t_PQij_bb = get_t_PQij(RPQRS_bb, kPQ_bb + fPQ_bb, kPQ_bb + fPQ_bb, nocc_b, nocc_b)
    t_PQij_ab = get_t_PQij(RPQRS_ab, kPQ_aa + fPQ_aa, kPQ_bb + fPQ_bb, nocc_a, nocc_b)

    def get_T(tau, RPQRS, t_PQij, nocc_m, nocc_n):
        T = tau.copy()
        T-= numpy.einsum('mnpq,pqij->mnij', RPQRS[:nocc_m,:nocc_n,:nmo,:nmo], t_PQij[:nmo,:nmo])
        T-= numpy.einsum('mncl,clij->mnij', RPQRS[:nocc_m,:nocc_n,nmo:,:nocc_n], t_PQij[nmo:,:nocc_n])
        T-= numpy.einsum('mnlc,lcij->mnij', RPQRS[:nocc_m,:nocc_n,:nocc_m,nmo:], t_PQij[:nocc_m,nmo:])
        return T
    
    T_aa = get_T(tau_aa, RPQRS_aa, t_PQij_aa, nocc_a, nocc_a)
    T_bb = get_T(tau_bb, RPQRS_bb, t_PQij_bb, nocc_b, nocc_b)
    T_ab = get_T(tau_ab, RPQRS_ab, t_PQij_ab, nocc_a, nocc_b)

    def get_p_PQij(RPQRS, kPQ_f1PQ_m, kPQ_f1PQ_n, nocc_m, nocc_n):
        p_PQij = numpy.einsum('RQij,RP->PQij', RPQRS[:,:,:nocc_m,:nocc_n], kPQ_f1PQ_m)
        p_PQij += numpy.einsum('PRij,RQ->PQij', RPQRS[:,:,:nocc_m,:nocc_n], kPQ_f1PQ_n)
        return p_PQij

    p_PQij_aa = get_p_PQij(RPQRS_aa, kPQ_aa + f1PQ_aa, kPQ_aa + f1PQ_aa, nocc_a, nocc_a)
    p_PQij_bb = get_p_PQij(RPQRS_bb, kPQ_bb + f1PQ_bb, kPQ_bb + f1PQ_bb, nocc_b, nocc_b)
    p_PQij_ab = get_p_PQij(RPQRS_ab, kPQ_aa + f1PQ_aa, kPQ_bb + f1PQ_bb, nocc_a, nocc_b)


    def get_P(RPQRS, p_PQij, nocc_m, nocc_n):
        P = numpy.einsum('mncd,cdij->mnij', RPQRS[:nocc_m,:nocc_n,nmo:,nmo:], p_PQij[nmo:,nmo:])
        P+= numpy.einsum('mnca,caij->mnij', RPQRS[:nocc_m,:nocc_n,nmo:,nocc_n:nmo], p_PQij[nmo:,nocc_n:nmo])
        P+= numpy.einsum('mnac,acij->mnij', RPQRS[:nocc_m,:nocc_n,nocc_m:nmo,nmo:], p_PQij[nocc_m:nmo,nmo:])
        return P

    P_aa = get_P(RPQRS_aa, p_PQij_aa, nocc_a, nocc_a)
    P_bb = get_P(RPQRS_bb, p_PQij_bb, nocc_b, nocc_b)
    P_ab = get_P(RPQRS_ab, p_PQij_ab, nocc_a, nocc_b)

    def get_q_PQij(RPQRS, kPQ_f1PQ_m, kPQ_f1PQ_n, nocc_m, nocc_n):
        q_PQij = numpy.einsum('PQRj,Ri->PQij', RPQRS[:,:,:,:nocc_n], kPQ_f1PQ_m[:,:nocc_m])
        q_PQij+= numpy.einsum('PQiR,Rj->PQij', RPQRS[:,:,:nocc_m,:], kPQ_f1PQ_n[:,:nocc_n])
        return q_PQij

    q_PQij_aa = get_q_PQij(RPQRS_aa, kPQ_aa + f1PQ_aa, kPQ_aa + f1PQ_aa, nocc_a, nocc_a)
    q_PQij_bb = get_q_PQij(RPQRS_bb, kPQ_bb + f1PQ_bb, kPQ_bb + f1PQ_bb, nocc_b, nocc_b)
    q_PQij_ab = get_q_PQij(RPQRS_ab, kPQ_aa + f1PQ_aa, kPQ_bb + f1PQ_bb, nocc_a, nocc_b)

    def get_Q(RbarmnRS, RPQRS, kPQ_f1PQ_m, kPQ_f1PQ_n, q_PQij, nocc_m, nocc_n):
        Q = numpy.einsum('mnPj,Pi->mnij',RbarmnRS[:nocc_m,:nocc_n,:,:nocc_n],kPQ_f1PQ_m[:,:nocc_m])
        Q+= numpy.einsum('mniP,Pj->mnij',RbarmnRS[:nocc_m,:nocc_n,:nocc_m,:],kPQ_f1PQ_n[:,:nocc_n])
        Q-= numpy.einsum('mnpq,pqij->mnij', RPQRS[:nocc_m,:nocc_n,:nmo,:nmo], q_PQij[:nmo,:nmo])
        Q-= numpy.einsum('mncl,clij->mnij', RPQRS[:nocc_m,:nocc_n,nmo:,:nocc_n], q_PQij[nmo:,:nocc_n])
        Q-= numpy.einsum('mnlc,lcij->mnij', RPQRS[:nocc_m,:nocc_n,:nocc_m,nmo:], q_PQij[:nocc_m,nmo:])
        return Q

    Q_aa = get_Q(RbarmnRS_aa, RPQRS_aa, kPQ_aa + f1PQ_aa, kPQ_aa + f1PQ_aa, q_PQij_aa, nocc_a, nocc_a)
    Q_bb = get_Q(RbarmnRS_bb, RPQRS_bb, kPQ_bb + f1PQ_bb, kPQ_bb + f1PQ_bb, q_PQij_bb, nocc_b, nocc_b)
    Q_ab = get_Q(RbarmnRS_ab, RPQRS_ab, kPQ_aa + f1PQ_aa, kPQ_bb + f1PQ_bb, q_PQij_ab, nocc_a, nocc_b)

    def get_C(RPQRS, f0PQ_m, f0PQ_n, nocc_m, nocc_n):
        C = numpy.einsum('mncb,ac->mnab',RPQRS[:nocc_m,:nocc_n,nmo:,nocc_n:nmo], f0PQ_m[nocc_m:nmo,nmo:])
        C+= numpy.einsum('mnac,bc->mnab',RPQRS[:nocc_m,:nocc_n,nocc_m:nmo,nmo:], f0PQ_n[nocc_n:nmo,nmo:])
        return C
    
    C_aa = get_C(RPQRS_aa, f0PQ_aa, f0PQ_aa, nocc_a, nocc_a)
    C_bb = get_C(RPQRS_bb, f0PQ_bb, f0PQ_bb, nocc_b, nocc_b)
    C_ab = get_C(RPQRS_ab, f0PQ_aa, f0PQ_bb, nocc_a, nocc_b)

    E_aa = numpy.einsum('mnab,abij->mnij',C_aa,RPQRS_aa[nocc_a:nmo,nocc_a:nmo,:nocc_a,:nocc_a])
    E_bb = numpy.einsum('mnab,abij->mnij',C_bb,RPQRS_bb[nocc_b:nmo,nocc_b:nmo,:nocc_b,:nocc_b])
    E_ab = numpy.einsum('mnab,abij->mnij',C_ab,RPQRS_ab[nocc_a:nmo,nocc_b:nmo,:nocc_a,:nocc_b])

    def B_contrib(X_mniP, X_mnPj, f0PQ_m, f0PQ_n, nocc_m, nocc_n, T, P, Q, E):
        B = numpy.einsum('mniP,Pj->mnij', X_mniP, f0PQ_n[:,:nocc_n])
        B += numpy.einsum('mnPj,Pi->mnij', X_mnPj, f0PQ_m[:,:nocc_m])
        B += T
        B -= P
        B += Q
        B -= E
        B = 0.5*(B + B.transpose(2,3,0,1))
        e = numpy.einsum('ijij->', B[frozen:,frozen:,frozen:,frozen:])
        return e
    
    emp2_f12 += B_contrib(X_mniP_aa, X_mnPj_aa, f0PQ_aa, f0PQ_aa, nocc_a, nocc_a, T_aa, P_aa, Q_aa, E_aa)
    emp2_f12 += B_contrib(X_mniP_bb, X_mnPj_bb, f0PQ_bb, f0PQ_bb, nocc_b, nocc_b, T_bb, P_bb, Q_bb, E_bb)
    emp2_f12 += B_contrib(X_mniP_ab, X_mnPj_ab, f0PQ_aa, f0PQ_bb, nocc_a, nocc_b, T_ab, P_ab, Q_ab, E_ab)
    print("f12 extra", emp2_f12)
    
    #CABS
    inv_e_ai_a = 1/(mo_energy[0][nocc_a:nmo,None] - mo_energy[0][None,:nocc_a])
    inv_e_ai_b = 1/(mo_energy[1][nocc_b:nmo,None] - mo_energy[1][None,:nocc_b])
    Pmo_energy_a = fPQ_aa.diagonal()
    Pmo_energy_b = fPQ_bb.diagonal()
    inv_e_ic_a = 1/(mo_energy[0][:nocc_a,None] - Pmo_energy_a[None,nmo:])
    inv_e_ic_b = 1/(mo_energy[1][:nocc_b,None] - Pmo_energy_b[None,nmo:])
    f_wig_a = fPQ_aa[:nocc_a,nmo:] - numpy.einsum("ac,ai,ia->ic", fPQ_aa[nocc_a:nmo,nmo:], inv_e_ai_a, fPQ_aa[:nocc_a, nocc_a:nmo])
    f_wig_b = fPQ_bb[:nocc_b,nmo:] - numpy.einsum("ac,ai,ia->ic", fPQ_bb[nocc_b:nmo,nmo:], inv_e_ai_b, fPQ_bb[:nocc_b, nocc_b:nmo])
    t_s_a = numpy.einsum("ic,ic->ic", f_wig_a, inv_e_ic_a)
    t_s_b = numpy.einsum("ic,ic->ic", f_wig_b, inv_e_ic_b)
    cabs_energy = numpy.einsum("ci,ic->", f_wig_a.T.conj(), t_s_a) + numpy.einsum("ci,ic->", f_wig_b.T.conj(), t_s_b)
    cabs_energy_new = numpy.inf
    print("init CABS energy", cabs_energy)
    adiis_a = lib.diis.DIIS()
    adiis_b = lib.diis.DIIS()
    icycle = 0
    diff_t_norm_a = diff_t_norm_b = 1.0
    def get_t_s_new(adiis, fPQ, t_s, inv_e_ai, inv_e_ic, f_wig, nmo, nocc):
        t_s_new = numpy.einsum("dc,id->ic", fPQ[nmo:,nmo:], t_s)
        t_s_new -= numpy.einsum("cc,ic->ic", fPQ[nmo:,nmo:], t_s)
        t_s_new -= numpy.einsum("ij,jc->ic", fPQ[:nocc,:nocc], t_s)
        t_s_new += numpy.einsum("ii,ic->ic", fPQ[:nocc,:nocc], t_s)
        t_s_new -= numpy.einsum("ac,ai,da,id->ic", fPQ[nocc:nmo,nmo:], inv_e_ai, fPQ[nmo:, nocc:nmo], t_s)
        t_s_new += f_wig
        t_s_new = numpy.einsum("ic,ic->ic", t_s_new, inv_e_ic)
        vec = amplitudes_to_vector(t_s_new)
        t_s_new = vector_to_amplitudes(adiis.update(vec), nocc)
        return t_s_new

    while (abs(cabs_energy - cabs_energy_new) > 1e-10 or diff_t_norm_a > 1e-10 or diff_t_norm_b > 1e-10) and icycle < 300:
        icycle += 1
        cabs_energy = cabs_energy_new
        t_s_new_a = get_t_s_new(adiis_a, fPQ_aa, t_s_a, inv_e_ai_a, inv_e_ic_a, f_wig_a, nmo, nocc_a)
        t_s_new_b = get_t_s_new(adiis_b, fPQ_bb, t_s_b, inv_e_ai_b, inv_e_ic_b, f_wig_b, nmo, nocc_b)
        diff_t_norm_a = numpy.linalg.norm(t_s_new_a - t_s_a)
        diff_t_norm_b = numpy.linalg.norm(t_s_new_b - t_s_b)
        t_s_a = t_s_new_a
        t_s_b = t_s_new_b
        cabs_energy_new = numpy.einsum("ci,ic->", f_wig_a.T.conj(), t_s_new_a) + numpy.einsum("ci,ic->", f_wig_b.T.conj(), t_s_new_b)
        print("CABS cycle", icycle, cabs_energy_new, abs(cabs_energy - cabs_energy_new), diff_t_norm_a, diff_t_norm_b)
    print("CABS energy", cabs_energy_new)
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

    mf = mf.to_uhf()

    mymp = ump2.UMP2(mf)
    e = mymp.kernel()[0]

    auxmol = mol.copy()
    auxmol.basis = "ccpvtz"
    #auxmol.basis = ('ccpvqz-fit', 'cc-pVQZ-F12-OptRI')
    #auxmol.basis = 'cc-pVQZ'
    auxmol.build(False, False)
    print('MP2', e)
    e+= energy_f12(mf, auxmol, 1., frozen=2)
    print('MP2-F12', e)
    print('e_tot', e+mf.e_tot)
