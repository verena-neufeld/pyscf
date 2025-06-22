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

"""
MP2-F12 (In testing)

Refs:
* JCC 32, 2492 (2011); DOI:10.1002/jcc.21825
* JCP 139, 084112 (2013); DOI:10.1063/1.4818753

Follows the second reference more closely.

With strong orthogonalization ansatz 2
"""

import warnings
from functools import reduce
import numpy
import scipy.linalg
from pyscf import lib, scf
from pyscf.lib import logger
from pyscf import gto
from pyscf.mp import mp2

warnings.warn("Module MP2-F12 is under testing")


def make_ghost_atoms(auxmol):
    # auxmol should provide auxilliary basis functions but not extra nuclei.
    # WARNING: this is a bit hacky. Check.
    for atm in auxmol._atm:
        atm[0] = 0  # remove charge
    atoms = []
    for atom in auxmol._atom:
        atoms.append(("GHOST-" + atom[0], atom[1]))
    auxmol._atom = atoms
    basis = {}
    for atm, bas in auxmol._basis.items():
        basis["GHOST-" + atm] = bas
    auxmol._basis = basis
    pseudo = {}
    for atm, ps in auxmol._pseudo.items():
        pseudo["GHOST-" + atm] = ps
    auxmol._pseudo = pseudo
    return auxmol


# The cabs space, the complimentary space to the OBS.
def find_cabs(nao, Pmf, lindep=1e-8):
    s = Pmf.get_ovlp()
    ls12 = scipy.linalg.solve(s[:nao, :nao], s[:nao, nao:], assume_a="her")
    s[nao:, nao:] -= s[nao:, :nao].dot(ls12)  # TODO: check conj()
    w, v = scipy.linalg.eigh(s[nao:, nao:])
    c2 = v[:, w > lindep] / numpy.sqrt(w[w > lindep])
    c1 = ls12.dot(c2)
    return numpy.vstack((-c1, c2))


def amplitudes_to_vector(t1, out=None):
    vector = t1.ravel()
    return vector


def vector_to_amplitudes(vector, nocc):
    naux = len(vector) // nocc
    t1 = vector.copy().reshape((nocc, naux))
    return t1


def get_h_k_f(Pmf, cabs_mol, assume_gbc, nmo, nocc):
    Pcoeff = Pmf.mo_coeff
    dm = Pmf.make_rdm1(Pcoeff, Pmf.mo_occ)
    vj, vk = Pmf.get_jk(cabs_mol, dm)
    kPQ = reduce(numpy.dot, (Pcoeff.conj().T, 0.5 * vk, Pcoeff))
    vhf = vj - vk * 0.5
    fockao = Pmf.get_fock(vhf=vhf, dm=dm)
    fPQ = reduce(numpy.dot, (Pcoeff.conj().T, fockao, Pcoeff))
    if assume_gbc:
        fPQ[:nmo, :nmo] = fPQ[:nmo, :nmo].diagonal() * numpy.eye(nmo)
        fPQ[:nocc, nocc:] = 0.0
        fPQ[nocc:, :nocc] = 0.0
    hPQ = fPQ.copy() + kPQ.copy()
    return hPQ, kPQ, fPQ


def get_tminj(nocc, frozen, zeta):
    tminj = numpy.zeros([nocc - frozen] * 4)
    for i in range(nocc - frozen):
        for j in range(nocc - frozen):
            tminj[i, i, j, j] = -3.0 / (8 * zeta)
            tminj[i, j, j, i] = -1.0 / (8 * zeta)
        tminj[i, i, i, i] = -0.5 / zeta
    return tminj


def energy_f12(mf, mp, auxmol, zeta, frozen=0, assume_gbc=False):
    # assume gbc means that f_ic=0 where i is occ and c is CABS.
    # if generalized Brillouin theorem is assumed,
    # Brillouin theorem is also assumed.
    logger.info(mf, "******** MP2-F12 (In testing) ********")
    auxmol = make_ghost_atoms(auxmol)
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    nocc = numpy.count_nonzero(mf.mo_occ == 2)
    nao, nmo = mo_coeff.shape

    cabs_mol = gto.conc_mol(mol, auxmol)

    Pmf = scf.RHF(cabs_mol)

    cabs_coeff = find_cabs(nao, Pmf)
    nca = cabs_coeff.shape[0]
    mo_o = mo_coeff[:, :nocc]
    mo_onf = mo_coeff[:, frozen:nocc]
    Pcoeff = numpy.vstack((mo_coeff, numpy.zeros((nca - nao, nmo))))
    Pcoeff = numpy.hstack((Pcoeff, cabs_coeff))

    Pmf.max_cycle = 0
    Pmf.mo_coeff = Pcoeff
    Pmf._eri = None
    Pmf.mo_occ = numpy.hstack((mf.mo_occ, numpy.zeros((Pcoeff.shape[1] - nmo))))

    obs = (0, mol.nbas)
    cbs = (0, cabs_mol.nbas)

    mol.set_f12_zeta(zeta)
    Y = mp.ao2mo_kernel(
        mol, "int2e_yp", [mo_onf] * 4, shls_slice=obs + obs + obs + obs, sla_zeta=zeta
    )
    cabs_mol.set_f12_zeta(zeta)
    RmPnQ = mp.ao2mo_kernel(
        cabs_mol,
        "int2e_stg",
        [mo_onf, Pcoeff, mo_onf, Pcoeff],
        shls_slice=obs + cbs + obs + cbs,
        sla_zeta=zeta,
    )
    Rmpnq = RmPnQ[:, :nmo, :, :nmo]
    Rmlnc = RmPnQ[:nocc, :nocc, :nocc, nmo:]
    Rmcnl = Rmlnc.transpose(2, 3, 0, 1)
    Rpiqj = Rmpnq.transpose(1, 0, 3, 2).copy().conj()
    Rlicj = Rmlnc.transpose(1, 0, 3, 2).copy().conj()
    Rcilj = Rlicj.transpose(2, 3, 0, 1)
    RRiQj = RmPnQ.transpose(1, 0, 3, 2).copy().conj()
    RmPnk = RmPnQ[:, :, :, :nocc]
    RQikj = RRiQj[:, :, :nocc, :]
    Rmknc = Rmlnc
    Rmpna = Rmpnq[:, :, :, nocc:nmo]
    Rqiaj = Rpiqj[:, :, nocc:nmo, :]
    RPicj = RRiQj[:, :, nmo:, :]
    Rmcnb = RmPnQ[:, nmo:, :, nocc:nmo]
    Rpibj = Rqiaj

    Rbar_miPj = mp.ao2mo_kernel(
        cabs_mol,
        "2int2e_stg",
        [Pcoeff, mo_onf, mo_onf, mo_onf],
        shls_slice=cbs + obs + obs + obs,
        sla_zeta=zeta,
    ).transpose(2, 3, 0, 1)
    Rbar_minj = Rbar_miPj[:, :, frozen:nocc, :].copy()
    # molecular calculation
    tau = Rbar_minj.copy() * zeta**2

    vpiqj = mp.ao2mo_kernel(
        mol,
        "int2e",
        [mo_coeff, mo_onf, mo_coeff, mo_onf],
        shls_slice=obs + obs + obs + obs,
    )
    vlicj = mp.ao2mo_kernel(
        cabs_mol,
        "int2e",
        [cabs_coeff, mo_onf, mo_o, mo_onf],
        shls_slice=cbs + obs + obs + obs,
    ).transpose(2, 3, 0, 1)
    vcilj = vlicj.transpose(2, 3, 0, 1)
    hPQ, kPQ, fPQ = get_h_k_f(Pmf, cabs_mol, assume_gbc, nmo, nocc)

    tminj = get_tminj(nocc, frozen, zeta)

    def add_V_contribution(m, n, Y, Rmpnq, vpiqj, Rmlnc, vlicj, Rmcnl, vcilj, tminj):
        # when m,n=i,j: V - iijj
        # when m,n=j,i: V - jiij
        V = lib.einsum(m + "i" + n + "j->ij", Y.copy())
        V -= lib.einsum(m + "p" + n + "q,piqj->ij", Rmpnq, vpiqj)
        V -= lib.einsum(m + "l" + n + "c,licj->ij", Rmlnc, vlicj)
        V -= lib.einsum(m + "c" + n + "l,cilj->ij", Rmcnl, vcilj)
        e = lib.einsum("ij," + m + "i" + n + "j", V, tminj) * 4
        e -= lib.einsum("ij," + n + "i" + m + "j", V, tminj) * 2
        e -= lib.einsum("ii,iiii", V, tminj)
        del V
        return e

    emp2_f12 = add_V_contribution("i", "j", Y, Rmpnq, vpiqj, Rmlnc, vlicj, Rmcnl,
                                  vcilj, tminj)
    emp2_f12 += add_V_contribution("j", "i", Y, Rmpnq, vpiqj, Rmlnc, vlicj, Rmcnl,
                                   vcilj, tminj)
    del Y

    def add_B_contribution(m, n, tau, Rbar_miPj, hPQ, frozen, nocc, RmPnQ, kPQ, fPQ,
                           RQikj, Rmknc, Rlicj, Rmpna, Rqiaj, Rmcnb, Rpibj, tminj):
        # when m,n=i,j: B - iijj
        # when m,n=j,i: B - jiij
        B = 0.5 * lib.einsum(m + "iPj," + n + "P->ij", Rbar_miPj, hPQ[frozen:nocc])
        B += 0.5 * lib.einsum(
            "i" + m + "P" + n + ",jP->ij", Rbar_miPj, hPQ[frozen:nocc]
        )
        tmp = lib.einsum(m + "P" + n + "Q,PR->" + m + "R" + n + "Q", RmPnQ, kPQ)
        B -= lib.einsum(m + "R" + n + "Q,RiQj->ij", tmp, RRiQj)
        tmp = lib.einsum(m + "P" + n + "k,PQ->" + m + "Q" + n + "k", RmPnk, fPQ)
        B -= lib.einsum(m + "Q" + n + "k,Qikj->ij", tmp, RQikj)
        tmp = lib.einsum(
            m + "k" + n + "c,kl->" + m + "l" + n + "c", Rmknc, fPQ[:nocc, :nocc]
        )
        B += lib.einsum(m + "l" + n + "c,licj->ij", tmp, Rlicj)
        tmp = lib.einsum(
            m + "p" + n + "a,pq->" + m + "q" + n + "a", Rmpna, fPQ[:nmo, :nmo]
        )
        B -= lib.einsum(m + "q" + n + "a,qiaj->ij", tmp, Rqiaj)
        tmp = lib.einsum(m + "k" + n + "c,kP->" + m + "P" + n + "c", Rmknc, fPQ[:nocc])
        tmp1 = lib.einsum(m + "P" + n + "c,Picj->ij", tmp, RPicj)
        tmp = lib.einsum(
            m + "c" + n + "b,cp->" + m + "p" + n + "b", Rmcnb, fPQ[nmo:, :nmo]
        )
        tmp1 += lib.einsum(m + "p" + n + "b,pibj->ij", tmp, Rpibj)
        if m == "i":
            B -= 2 * tmp1
        else:
            B -= tmp1 + tmp1.transpose(1, 0)
        B = B + B.transpose(1, 0)
        B += lib.einsum(m + "i" + n + "j->ij", tau.copy())
        tmp = lib.einsum("o" + m + "k" + n + ",ij->oikj", tminj, B)
        e = lib.einsum("oikj,oikj", tmp, tminj) * 2
        e -= lib.einsum("oikj,kioj", tmp, tminj)
        # need to subtract half of iiii contribution since both iijj and jiij contribute that.
        e -= lib.einsum("iiii,iiii", tmp, tminj) * 0.5
        return e

    emp2_f12 += add_B_contribution("i", "j", tau, Rbar_miPj, hPQ, frozen, nocc, RmPnQ, kPQ, fPQ,
                           RQikj, Rmknc, Rlicj, Rmpna, Rqiaj, Rmcnb, Rpibj, tminj)
    emp2_f12 += add_B_contribution("j", "i", tau, Rbar_miPj, hPQ, frozen, nocc, RmPnQ, kPQ, fPQ,
                           RQikj, Rmknc, Rlicj, Rmpna, Rqiaj, Rmcnb, Rpibj, tminj)
    del tau

    X_dict = {}
    Xtmp_dict = {}

    def get_X(o, l, Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj):
        # returning X_oilj where one of o or l equals i or j.
        if o in ["i", "j"]:
            assert l == "k"
            xstring = "i" + l + "j"
        elif l in ["i", "j"]:
            assert o == "k"
            xstring = o + "ij"
        else:
            raise RuntimeError
        ext_xstring = o + "i" + l + "j"
        if ext_xstring in X_dict.keys():
            X = X_dict[ext_xstring]
        else:
            X = lib.einsum(o + "i" + l + "j->" + xstring, Rbar_minj.copy())
            X -= lib.einsum(o + "p" + l + "q,piqj->" + xstring, Rmpnq, Rpiqj)
            X -= lib.einsum(o + "z" + l + "c,zicj->" + xstring, Rmlnc, Rlicj)
            X -= lib.einsum(o + "c" + l + "z,cizj->" + xstring, Rmcnl, Rcilj)
            X_dict[ext_xstring] = X
        return X, xstring

    def get_Xtmp(m, n, Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj, nocc, frozen, tminj):
        if m in ["i", "j"]:
            assert n == "k"
            xtmpstring = "i" + n + "j"
        elif n in ["i", "j"]:
            assert m == "k"
            xtmpstring = m + "ij"
        else:
            raise RuntimeError
        ext_xtmpstring = m + "i" + n + "j"
        if ext_xtmpstring in Xtmp_dict.keys():
            tmp = Xtmp_dict[ext_xtmpstring]
        else:
            # m = o, n = l
            X, xstring = get_X(m, n, Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj)
            tmp = lib.einsum(
                m + m + n + n + "," + xstring + "->" + xtmpstring, tminj, X
            )
            # need to subtract one case of m = l = o = n
            if n == "i" or m == "i":
                # xtmpstring is iij when m=n.
                for i in range(nocc - frozen):
                    for j in range(nocc - frozen):
                        tmp[i, i, j] = 0.0
            elif n == "j":
                # xtmpstring is jij when m=n.
                for i in range(nocc - frozen):
                    for j in range(nocc - frozen):
                        tmp[j, i, j] = 0.0
            elif m == "j":
                # xtmpstring is ijj when m=n.
                for i in range(nocc - frozen):
                    for j in range(nocc - frozen):
                        tmp[i, j, j] = 0.0
            else:
                raise RuntimeError
            # m = l, o = n
            X, xstring = get_X(n, m, Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj)
            tmp += lib.einsum(
                m + n + n + m + "," + xstring + "->" + xtmpstring, tminj, X
            )
            Xtmp_dict[ext_xtmpstring] = tmp
        return tmp, xtmpstring

    def add_X_contribution(m, n, Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj, nocc,
                           frozen, tminj):
        # m, n = i, j or m, n = j, i
        xtmp, xtmpstring = get_Xtmp("k", n, Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj,
                                    nocc, frozen, tminj)
        e = (
            -lib.einsum(
                m + "k," + xtmpstring + "," + m + "i" + n + "j",
                fPQ[frozen:nocc, frozen:nocc],
                xtmp,
                tminj,
            )
            * 2
        )
        e += lib.einsum(
            m + "k," + xtmpstring + "," + n + "i" + m + "j",
            fPQ[frozen:nocc, frozen:nocc],
            xtmp,
            tminj,
        )
        # both combination will have a j = i = n = m contribution, so substract half.
        e += lib.einsum("ik,kii,iiii", fPQ[frozen:nocc, frozen:nocc], xtmp, tminj) * 0.5

        xtmp, xtmpstring = get_Xtmp(m, "k", Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj,
                                    nocc, frozen, tminj)
        e -= (
            lib.einsum(
                "k" + n + "," + xtmpstring + "," + m + "i" + n + "j",
                fPQ[frozen:nocc, frozen:nocc],
                xtmp,
                tminj,
            )
            * 2
        )
        e += lib.einsum(
            "k" + n + "," + xtmpstring + "," + n + "i" + m + "j",
            fPQ[frozen:nocc, frozen:nocc],
            xtmp,
            tminj,
        )
        # both combination will have a j = i = n = m contribution, so substract half.
        e += lib.einsum("ki,iki,iiii", fPQ[frozen:nocc, frozen:nocc], xtmp, tminj) * 0.5
        return e

    emp2_f12 += add_X_contribution("i", "j", Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj,
                                    nocc, frozen, tminj)
    emp2_f12 += add_X_contribution("j", "i", Rbar_minj, Rmpnq, Rpiqj, Rmlnc, Rlicj, Rmcnl, Rcilj,
                                    nocc, frozen, tminj)

    cabs_singles = get_cabs_singles(mf.mo_energy, fPQ, nocc, nmo)
    print("CABS singles", cabs_singles)
    print("return f12 extra (without CABS)", emp2_f12)
    return emp2_f12, cabs_singles


def get_cabs_singles(mo_energy, fPQ, nocc, nmo):
    inv_e_ai = 1 / (mo_energy[nocc:nmo, None] - mo_energy[None, :nocc])
    Pmo_energy = fPQ.diagonal()
    inv_e_ic = 1 / (mo_energy[:nocc, None] - Pmo_energy[None, nmo:])
    f_wig = fPQ[:nocc, nmo:] - lib.einsum(
        "ac,ai,ia->ic", fPQ[nocc:nmo, nmo:], inv_e_ai, fPQ[:nocc, nocc:nmo]
    )
    t_s = lib.einsum("ic,ic->ic", f_wig, inv_e_ic)
    cabs_energy = 2.0 * lib.einsum("ci,ic->", f_wig.T.conj(), t_s)
    cabs_energy_new = numpy.inf
    print("init CABS energy", cabs_energy)
    adiis = lib.diis.DIIS()
    adiis.space = 10
    icycle = 0
    diff_t_norm = 1.0
    while (
        abs(cabs_energy - cabs_energy_new) > 1e-10 or diff_t_norm > 1e-10
    ) and icycle < 300:
        icycle += 1
        cabs_energy = cabs_energy_new
        t_s_new = lib.einsum("dc,id->ic", fPQ[nmo:, nmo:], t_s)
        t_s_new -= lib.einsum("cc,ic->ic", fPQ[nmo:, nmo:], t_s)
        t_s_new -= lib.einsum("ij,jc->ic", fPQ[:nocc, :nocc], t_s)
        t_s_new += lib.einsum("ii,ic->ic", fPQ[:nocc, :nocc], t_s)
        t_s_new -= lib.einsum(
            "ac,ai,da,id->ic", fPQ[nocc:nmo, nmo:], inv_e_ai, fPQ[nmo:, nocc:nmo], t_s
        )
        t_s_new += f_wig
        t_s_new = lib.einsum("ic,ic->ic", t_s_new, inv_e_ic)
        vec = amplitudes_to_vector(t_s_new)
        t_s_new = vector_to_amplitudes(adiis.update(vec), nocc)
        diff_t_norm = numpy.linalg.norm(t_s_new - t_s)
        t_s = t_s_new
        cabs_energy_new = 2.0 * lib.einsum("ci,ic->", f_wig.T.conj(), t_s_new)
        print(
            "CABS cycle",
            icycle,
            cabs_energy_new,
            abs(cabs_energy - cabs_energy_new),
            diff_t_norm,
        )
    # Check whether R in eq. 35 (Turbomole paper) is really zero.
    tmp = f_wig + lib.einsum("dc,id->ic", fPQ[nmo:, nmo:], t_s)
    tmp -= lib.einsum("ij,jc->ic", fPQ[:nocc, :nocc], t_s)
    tmp -= lib.einsum(
        "ac,ai,da,id->ic", fPQ[nocc:nmo, nmo:], inv_e_ai, fPQ[nmo:, nocc:nmo], t_s
    )
    if numpy.max(numpy.abs(tmp)) < 1e-8:
        print("CABS SUCCESS. max R", numpy.max(numpy.abs(tmp)))
    else:
        print("CABS FAIL. max R", numpy.max(numpy.abs(tmp)))
    print("CABS energy", cabs_energy_new)
    return cabs_energy_new


if __name__ == "__main__":
    bas = "d"
    b = 1.8
    from pyscf import scf, df
    import time
    import pandas as pd

    mol = gto.Mole()
    mol.atom = "Ne 0 0 0;"
    print(mol.atom)
    mol.basis = "augccpv" + bas + "z"
    mol.build()
    mf = scf.RHF(mol)
    # mf.max_cycle = 2
    t0 = time.time()
    print(mf.scf())
    print("HF time", time.time() - t0)

    mymp = mp2.MP2(mf)
    e = mymp.kernel()[0]

    auxmol = mol.copy()
    auxmol.basis = {"Ne": "aug-cc-pv" + bas + "z-optri_ne.0.nw"}
    # auxmol.basis = ('ccpvqz-fit', 'cc-pVQZ-F12-OptRI')
    # auxmol.basis = 'cc-pVQZ'
    auxmol.build(False, False)
    print("MP2", e)
    t1 = time.time()
    e_f12 = energy_f12(mf, mymp, auxmol, 1.5, frozen=1)
    difft = time.time() - t1
    e += e_f12
    print("MP2-F12", e)
    print("e_tot", e + mf.e_tot)
    print("e f12", e_f12, "time taken", difft)
