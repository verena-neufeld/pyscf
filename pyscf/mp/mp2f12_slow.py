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

Follows the second reference more closely.

With strong orthogonalization ansatz 2
'''

import warnings
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
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
    pseudo = {}
    for atm, ps in auxmol._pseudo.items():
        pseudo["GHOST-"+atm] = ps
    auxmol._pseudo = pseudo
    return auxmol

# The cabs space, the complimentary space to the OBS.
def find_cabs(nao, Pmf, lindep=1e-8):
    s = Pmf.get_ovlp()
    ls12 = scipy.linalg.solve(s[:nao,:nao], s[:nao,nao:], assume_a='her')
    s[nao:,nao:] -= s[nao:,:nao].dot(ls12) # check conj()
    w, v = scipy.linalg.eigh(s[nao:,nao:])
    c2 = v[:,w>lindep]/numpy.sqrt(w[w>lindep])
    c1 = ls12.dot(c2)
    return numpy.vstack((-c1,c2))

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
    nocc = numpy.count_nonzero(mf.mo_occ == 2)
    nao, nmo = mo_coeff.shape
   
    if hasattr(mol, "dimension"):
        from pyscf.pbc import gto as pbcgto
        cabs_mol = pbcgto.conc_cell(mol, auxmol)
    else:
        cabs_mol = gto.conc_mol(mol, auxmol)
   
    if hasattr(mol, "dimension"):
        from pyscf.pbc import scf as pbcscf
        Pmf = pbcscf.RHF(cabs_mol, kpt=mf.kpt).density_fit()
    else:
        from pyscf import scf
        Pmf = scf.RHF(cabs_mol)

    cabs_coeff = find_cabs(nao, Pmf)
    nca = cabs_coeff.shape[0]
    mo_o = mo_coeff[:,:nocc]
    mo_onf = mo_coeff[:,frozen:nocc]
    Pcoeff = numpy.vstack((mo_coeff, numpy.zeros((nca-nao, nmo))))
    Pcoeff = numpy.hstack((Pcoeff, cabs_coeff))

    Pmf.max_cycle = 0
    #Pmf.kernel()
    Pmf.mo_coeff = Pcoeff
    Pmf._eri = None
    Pmf.mo_occ = numpy.hstack((mf.mo_occ, numpy.zeros((Pcoeff.shape[1] - nmo))))

    obs = (0, mol.nbas)
    cbs = (0, cabs_mol.nbas)
    
    mol.set_f12_zeta(zeta)
    Y = mp.ao2mo_kernel(mol, 'int2e_yp', [mo_onf]*4, shls_slice=obs+obs+obs+obs, sla_zeta=zeta)

    cabs_mol.set_f12_zeta(zeta)
    RmPnQ = mp.ao2mo_kernel(cabs_mol, 'int2e_stg', [mo_onf, Pcoeff, mo_onf, Pcoeff], shls_slice=obs+cbs+obs+cbs, sla_zeta=zeta)
    Rmpnq = RmPnQ[:,:nmo,:,:nmo]
    Rmlnc = RmPnQ[:nocc,:nocc,:nocc,nmo:]
    Rmcnl = Rmlnc.transpose(2,3,0,1)
    Rpiqj = Rmpnq.transpose(1,0,3,2).copy().conj()
    Rlicj = Rmlnc.transpose(1,0,3,2).copy().conj()
    Rcilj = Rlicj.transpose(2,3,0,1)
    RRiQj = RmPnQ.transpose(1,0,3,2).copy().conj()
    RmPnk = RmPnQ[:,:,:,:nocc]
    RQikj = RRiQj[:,:,:nocc,:]
    Rmknc = Rmlnc
    Rmpna = Rmpnq[:,:,:,nocc:nmo]
    Rqiaj = Rpiqj[:,:,nocc:nmo,:]
    RPicj = RRiQj[:,:,nmo:,:]
    Rmcnb = RmPnQ[:,nmo:,:,nocc:nmo]
    Rpibj = Rqiaj

    Rbar_miPj = mp.ao2mo_kernel(cabs_mol, '2int2e_stg', [Pcoeff, mo_onf, mo_onf, mo_onf], shls_slice=cbs+obs+obs+obs, sla_zeta=zeta).transpose(2,3,0,1)
    Rbar_minj = Rbar_miPj[:,:,frozen:nocc,:].copy()
    if hasattr(mol, "dimension"):
        # supercell calculation
        tau = mp.ao2mo_kernel(mol, 'gradint2e_stg', [mo_onf]*4, shls_slice=obs+obs+obs+obs, sla_zeta=zeta)
    else:
        # molecular calculation
        tau = Rbar_minj.copy() * zeta**2

    vpiqj = mp.ao2mo_kernel(mol, 'int2e', [mo_coeff, mo_onf, mo_coeff, mo_onf], shls_slice=obs+obs+obs+obs)
    vlicj = mp.ao2mo_kernel(cabs_mol, 'int2e', [cabs_coeff, mo_onf, mo_o, mo_onf], shls_slice=cbs+obs+obs+obs).transpose(2,3,0,1)
    vcilj = vlicj.transpose(2,3,0,1)

    dm = Pmf.make_rdm1(Pcoeff, Pmf.mo_occ)
    vj, vk = Pmf.get_jk(cabs_mol, numpy.asarray(dm))
    kPQ = reduce(numpy.dot, (Pcoeff.conj().T, 0.5*vk, Pcoeff))
    vhf =  vj - vk * .5 
    fockao = Pmf.get_fock(vhf=vhf, dm=dm)
    fPQ = reduce(numpy.dot, (Pcoeff.conj().T, fockao, Pcoeff))
    hPQ = fPQ.copy() + kPQ.copy()

    tminj = numpy.zeros([nocc-frozen]*4)
    for i in range(nocc-frozen):
        for j in range(nocc-frozen):
            tminj[i,i,j,j] = -3./(8*zeta)
            tminj[i,j,j,i] = -1./(8*zeta)
        tminj[i,i,i,i] = -.5/zeta

    V = Y.copy()
    V-= numpy.einsum('mpnq,piqj->minj', Rmpnq, vpiqj)
    V-= numpy.einsum('mlnc,licj->minj', Rmlnc, vlicj)
    V-= numpy.einsum('mcnl,cilj->minj', Rmcnl, vcilj)
    emp2_f12 = numpy.einsum('minj,minj', V, tminj) * 4
    emp2_f12-= numpy.einsum('minj,nimj', V, tminj) * 2
    
    X = Rbar_minj.copy()
    X-= numpy.einsum('mpnq,piqj->minj', Rmpnq, Rpiqj)
    X-= numpy.einsum('mlnc,licj->minj', Rmlnc, Rlicj)
    X-= numpy.einsum('mcnl,cilj->minj', Rmcnl, Rcilj)

    tmp = numpy.einsum('miPj,nP->minj', Rbar_miPj, hPQ[frozen:nocc])
    B   = (tmp + tmp.transpose(1,0,3,2)) * .5
    tmp = numpy.einsum('mPnQ,PR->mRnQ', RmPnQ, kPQ)
    B  -= numpy.einsum('mRnQ,RiQj->minj', tmp, RRiQj)
    tmp = numpy.einsum('mPnk,PQ->mQnk', RmPnk, fPQ)
    B  -= numpy.einsum('mQnk,Qikj->minj', tmp, RQikj)
    tmp = numpy.einsum('mknc,kl->mlnc', Rmknc, fPQ[:nocc,:nocc])
    B  += numpy.einsum('mlnc,licj->minj', tmp, Rlicj)
    tmp = numpy.einsum('mpna,pq->mqna', Rmpna, fPQ[:nmo,:nmo])
    B  -= numpy.einsum('mqna,qiaj->minj', tmp, Rqiaj)
    tmp = numpy.einsum('mknc,kP->mPnc', Rmknc, fPQ[:nocc])
    tmp1= numpy.einsum('mPnc,Picj->minj', tmp, RPicj)
    tmp = numpy.einsum('mcnb,cp->mpnb', Rmcnb, fPQ[nmo:,:nmo])
    tmp1+= numpy.einsum('mpnb,pibj->minj', tmp, Rpibj)
    B  -= tmp1 + tmp1.transpose(1,0,3,2)
    B   = B + B.transpose(2,3,0,1)
    B  += tau
    tmp = numpy.einsum('mknl,kilj->minj', tminj, B)
    emp2_f12+= numpy.einsum('minj,minj', tmp, tminj) * 2
    emp2_f12-= numpy.einsum('minj,nimj', tmp, tminj)
    
    tmp = numpy.einsum('mknl,kilj->minj', tminj, X)
    emp2_f12-= numpy.einsum('mk,kinj,minj', fPQ[frozen:nocc,frozen:nocc], tmp, tminj) * 2
    emp2_f12+= numpy.einsum('mk,kinj,nimj', fPQ[frozen:nocc,frozen:nocc], tmp, tminj)
    emp2_f12-= numpy.einsum('kn,mikj,minj', fPQ[frozen:nocc,frozen:nocc], tmp, tminj) * 2
    emp2_f12+= numpy.einsum('kn,mikj,nimj', fPQ[frozen:nocc,frozen:nocc], tmp, tminj)

    print("return f12 extra (without CABS)", emp2_f12)
    return emp2_f12


if __name__ == '__main__':
    bas = "q"
    from pyscf import scf
    import time

    mol = gto.Mole()
    mol.atom = 'C 0 0 0; C 0 0 2'
    mol.basis = "augccpv"+bas+"z"
    mol.build()
    mf = scf.RHF(mol)
    #mf.max_cycle = 2
    print(mf.scf())

    mymp = mp2.MP2(mf)
    e = mymp.kernel()[0]

    auxmol = mol.copy()
    auxmol.basis = {"C": "aug-cc-pv"+bas+"z-optri.0.nw"}
    #auxmol.basis = ('ccpvqz-fit', 'cc-pVQZ-F12-OptRI')
    #auxmol.basis = 'cc-pVQZ'
    auxmol.build(False, False)
    print('MP2', e)
    t1 = time.time()
    e_f12= energy_f12(mf, mymp, auxmol, 1., frozen=2)
    difft = time.time()-t1
    e+=e_f12
    print('MP2-F12', e)
    print('e_tot', e+mf.e_tot)
    print('e f12', e_f12, "time taken", difft)
