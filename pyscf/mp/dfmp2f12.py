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
import ctypes
import numpy
import scipy.linalg
from pyscf import lib, __config__, scf
from pyscf.lib import logger
from pyscf.ao2mo import _ao2mo
from pyscf import gto
from pyscf.mp import mp2, dfmp2, mp2f12

warnings.warn("Module MP2-F12 is under testing")
libmp = lib.load_library("libmp")
THRESH_LINDEP = getattr(__config__, "mp_dfmp2_thresh_lindep", 1e-10)


def energy_f12(mf, mp, auxmol, df_auxmol, zeta, frozen=0, assume_gbc=False):
    # assume gbc means that f_ic=0 where i is occ and c is CABS.
    # if generalized Brillouin theorem is assumed,
    # Brillouin theorem is also assumed.
    logger.info(mf, "******** MP2-F12 (In testing) ********")
    auxmol = mp2f12.make_ghost_atoms(auxmol)
    mol = mf.mol
    mo_coeff = mf.mo_coeff
    nocc = numpy.count_nonzero(mf.mo_occ == 2)
    nao, nmo = mo_coeff.shape

    cabs_mol = gto.conc_mol(mol, auxmol)

    Pmf = scf.RHF(cabs_mol).density_fit(auxbasis=df_auxmol.basis)

    cabs_coeff = mp2f12.find_cabs(nao, Pmf)
    nca = cabs_coeff.shape[0]
    mo_onf = mo_coeff[:, frozen:nocc]
    Pcoeff = numpy.vstack((mo_coeff, numpy.zeros((nca - nao, nmo))))
    Pcoeff = numpy.hstack((Pcoeff, cabs_coeff))

    Pmf.max_cycle = 0
    Pmf.mo_coeff = Pcoeff
    Pmf._eri = None
    Pmf.mo_occ = numpy.hstack((mf.mo_occ, numpy.zeros((Pcoeff.shape[1] - nmo))))

    mol.set_f12_zeta(zeta)
    cabs_mol.set_f12_zeta(zeta)

    with_df = DF_aux(df_auxmol, mp, zeta)
    max_memory = 500000000000
    RiP = numpy.asarray(
        _init_mp_df_eris_direct(
            with_df,
            cabs_mol,
            Pcoeff[:, frozen:nocc],
            Pcoeff,
            max_memory,
            int_type_suff="_stg_sph",
        ),
        order="C",
    ).reshape(nocc - frozen, Pcoeff.shape[1], -1)
    Ymi = numpy.asarray(
        _init_mp_df_eris_direct(
            with_df, mol, mo_onf, mo_onf, max_memory, int_type_suff="_yp_sph"
        ),
        order="C",
    ).reshape(nocc - frozen, nocc - frozen, -1)
    cabs_mol.set_f12_zeta(2 * zeta)
    with_df.auxmol.set_f12_zeta(2 * zeta)
    RbarPi = numpy.asarray(
        _init_mp_df_eris_direct(
            with_df,
            cabs_mol,
            Pcoeff,
            Pcoeff[:, frozen:nocc],
            max_memory,
            int_type_suff="_stg_sph",
        ),
        order="C",
    ).reshape(Pcoeff.shape[1], nocc - frozen, -1)
    cabs_mol.set_f12_zeta(zeta)
    with_df.auxmol.set_f12_zeta(zeta)

    RmPnQ = lib.einsum("iPX,jQX->iPjQ", RiP, RiP)

    hPQ, kPQ, fPQ = mp2f12.get_h_k_f(Pmf, cabs_mol, assume_gbc, nmo, nocc)

    tminj = tminj = mp2f12.get_tminj(nocc, frozen, zeta)

    GPi2 = dfmp2._init_mp_df_eris(
        Pmf.with_df, Pcoeff, Pcoeff[:, frozen:nocc], max_memory
    ).reshape(Pcoeff.shape[1], nocc - frozen, -1)
    vpiqj = lib.einsum("piX,qjX->piqj", GPi2[:nmo, :, :], GPi2[:nmo, :, :])
    vlicj = lib.einsum("liX,cjX->licj", GPi2[:nocc, :, :], GPi2[nmo:, :, :])
    del GPi2

    def add_V_contribution(zeta, Ymi, RmPnQ, nocc, nmo, vpiqj, vlicj):
        e = 0.0
        fac_iijj = 2 * -3.0 / (8 * zeta) * 4 - 2 * -1.0 / (8 * zeta) * 2
        fac_jiij = 2 * -1.0 / (8 * zeta) * 4 - 2 * -3.0 / (8 * zeta) * 2
        for i in range(nocc - frozen):
            for j in range(i + 1, nocc - frozen):
                V = fac_iijj * numpy.dot(Ymi[i, i], Ymi[j, j]) + fac_jiij * numpy.dot(
                    Ymi[j, i], Ymi[i, j]
                )
                V -= lib.einsum(
                    "pq,pq->",
                    RmPnQ[i, :nmo, j, :nmo],
                    fac_iijj * vpiqj[:, i, :, j] + fac_jiij * vpiqj[:, j, :, i],
                )
                V -= lib.einsum(
                    "lc,lc->",
                    RmPnQ[i, :nocc, j, nmo:],
                    fac_iijj * vlicj[:, i, :, j] + fac_jiij * vlicj[:, j, :, i],
                )
                V -= lib.einsum(
                    "cl,cl->",
                    RmPnQ[i, nmo:, j, :nocc],
                    fac_iijj * vlicj[:, j, :, i].T + fac_jiij * vlicj[:, i, :, j].T,
                )
                e += V
            V = numpy.dot(Ymi[i, i], Ymi[i, i])
            V -= lib.einsum("pq,pq->", RmPnQ[i, :nmo, i, :nmo], vpiqj[:, i, :, i])
            V -= lib.einsum("lc,lc->", RmPnQ[i, :nocc, i, nmo:], vlicj[:, i, :, i])
            V -= lib.einsum(
                "cl,cl->",
                RmPnQ[i, nmo:, i, :nocc],
                vlicj.transpose(2, 3, 0, 1)[:, i, :, i],
            )
            e += -0.5 / zeta * V * 2
        return e

    emp2_f12 = add_V_contribution(zeta, Ymi, RmPnQ, nocc, nmo, vpiqj, vlicj)

    del vpiqj, vlicj, Ymi

    RRiQj = RmPnQ.transpose(1, 0, 3, 2).copy().conj()
    RiP_kPQ = lib.einsum("iPX,PQ->iQX", RiP, kPQ)
    RmPnQ_kPQ = lib.einsum("iPX,jQX->iPjQ", RiP_kPQ, RiP)
    RiP_fPQ = lib.einsum("iPX,PQ->iQX", RiP, fPQ)
    RmPnk_fPQ = lib.einsum("iPX,jkX->iPjk", RiP_fPQ, RiP[:, :nocc, :])
    del RiP, RiP_kPQ
    Rbar_minj = lib.einsum(
        "miX,njX->minj", RbarPi[frozen:nocc, :, :], RbarPi[frozen:nocc, :, :]
    )
    tau = Rbar_minj.copy() * zeta**2
    Rbar_miPj = lib.einsum("miX,PjX->miPj", RbarPi[frozen:nocc, :, :], RbarPi)

    del RbarPi

    def add_B_contribution(zeta, nocc, frozen, Rbar_miPj, hPQ, RmPnQ_kPQ, RRiQj,
                           fPQ, tau, RmPnk_fPQ, RmPnQ):
        e = 0.0
        fac_mi = 2 * (
            2 * (-3.0 / (8 * zeta)) * (-3.0 / (8 * zeta))
            - (-3.0 / (8 * zeta)) * (-1.0 / (8 * zeta))
            + 2 * (-1.0 / (8 * zeta)) * (-1.0 / (8 * zeta))
            - (-1.0 / (8 * zeta)) * (-3.0 / (8 * zeta))
        )
        fac_mj = 2 * (
            2 * (-1.0 / (8 * zeta)) * (-3.0 / (8 * zeta))
            - (-1.0 / (8 * zeta)) * (-1.0 / (8 * zeta))
            + 2 * (-3.0 / (8 * zeta)) * (-1.0 / (8 * zeta))
            - (-3.0 / (8 * zeta)) * (-3.0 / (8 * zeta))
        )
        for i in range(nocc - frozen):
            for j in range(nocc - frozen):
                if i == j:
                    continue
                B = lib.einsum(
                    "P,P->", fac_mi * Rbar_miPj[i, i, :, j], hPQ[j + frozen, :]
                ) + fac_mj * lib.einsum(
                    "P,P->", Rbar_miPj[j, i, :, j], hPQ[i + frozen, :]
                )
                B -= lib.einsum(
                    "RQ,RQ->",
                    fac_mi * RmPnQ_kPQ[i, :, j, :] + fac_mj * RmPnQ_kPQ[j, :, i, :],
                    RRiQj[:, i, :, j],
                )
                B -= lib.einsum(
                    "Qk,Qk->",
                    fac_mi * RmPnk_fPQ[i, :, j, :] + fac_mj * RmPnk_fPQ[j, :, i, :],
                    RRiQj[:, i, :nocc, j],
                )
                if assume_gbc:
                    tmp = lib.einsum(
                        "lc,ll->lc",
                        fac_mi * RmPnQ[i, :nocc, j, nmo:]
                        + fac_mj * RmPnQ[j, :nocc, i, nmo:],
                        fPQ[:nocc, :nocc],
                    )
                else:
                    tmp = lib.einsum(
                        "kc,kl->lc",
                        fac_mi * RmPnQ[i, :nocc, j, nmo:]
                        + fac_mj * RmPnQ[j, :nocc, i, nmo:],
                        fPQ[:nocc, :nocc],
                    )
                B += lib.einsum("lc,lc->", tmp, RRiQj[:nocc, i, nmo:, j])
                if assume_gbc:
                    tmp = lib.einsum(
                        "qa,qq->qa",
                        fac_mi * RmPnQ[i, :nmo, j, nocc:nmo]
                        + fac_mj * RmPnQ[j, :nmo, i, nocc:nmo],
                        fPQ[:nmo, :nmo],
                    )
                else:
                    tmp = lib.einsum(
                        "pa,pq->qa",
                        fac_mi * RmPnQ[i, :nmo, j, nocc:nmo]
                        + fac_mj * RmPnQ[j, :nmo, i, nocc:nmo],
                        fPQ[:nmo, :nmo],
                    )
                B -= lib.einsum("qa,qa->", tmp, RRiQj[:nmo, i, nocc:nmo, j])
                if assume_gbc:
                    tmp = lib.einsum(
                        "kc,kk->kc",
                        fac_mi * RmPnQ[i, :nocc, j, nmo:]
                        + fac_mj * RmPnQ[j, :nocc, i, nmo:],
                        fPQ[:nocc, :nocc],
                    )
                    B -= 2 * lib.einsum("kc,kc->", tmp, RRiQj[:nocc, i, nmo:, j])
                else:
                    tmp = lib.einsum(
                        "kc,kP->Pc",
                        fac_mi * RmPnQ[i, :nocc, j, nmo:]
                        + fac_mj * RmPnQ[j, :nocc, i, nmo:],
                        fPQ[:nocc],
                    )
                    B -= 2 * lib.einsum("Pc,Pc->", tmp, RRiQj[:, i, nmo:, j])
                if assume_gbc:
                    tmp = lib.einsum(
                        "cb,ca->ab",
                        fac_mi * RmPnQ[i, nmo:, j, nocc:nmo]
                        + fac_mj * RmPnQ[j, nmo:, i, nocc:nmo],
                        fPQ[nmo:, nocc:nmo],
                    )
                    B -= 2 * lib.einsum("ab,ab->", tmp, RRiQj[nocc:nmo, i, nocc:nmo, j])
                else:
                    tmp = lib.einsum(
                        "cb,cp->pb",
                        fac_mi * RmPnQ[i, nmo:, j, nocc:nmo]
                        + fac_mj * RmPnQ[j, nmo:, i, nocc:nmo],
                        fPQ[nmo:, :nmo],
                    )
                    B -= 2 * lib.einsum("pb,pb->", tmp, RRiQj[:nmo, i, nocc:nmo, j])
                B += 0.5 * fac_mi * tau[i, i, j, j] + 0.5 * fac_mj * tau[j, i, i, j]
                e += B
            B = lib.einsum("P,P->", Rbar_miPj[i, i, :, i], hPQ[i + frozen, :])
            B -= lib.einsum("RQ,RQ->", RmPnQ_kPQ[i, :, i, :], RRiQj[:, i, :, i])
            B -= lib.einsum("Qk,Qk->", RmPnk_fPQ[i, :, i, :], RRiQj[:, i, :nocc, i])
            if assume_gbc:
                tmp = lib.einsum(
                    "lc,ll->lc", RmPnQ[i, :nocc, i, nmo:], fPQ[:nocc, :nocc]
                )
            else:
                tmp = lib.einsum(
                    "kc,kl->lc", RmPnQ[i, :nocc, i, nmo:], fPQ[:nocc, :nocc]
                )
            B += lib.einsum("lc,lc->", tmp, RRiQj[:nocc, i, nmo:, i])
            if assume_gbc:
                tmp = lib.einsum(
                    "qa,qq->qa", RmPnQ[i, :nmo, i, nocc:nmo], fPQ[:nmo, :nmo]
                )
            else:
                tmp = lib.einsum(
                    "pa,pq->qa", RmPnQ[i, :nmo, i, nocc:nmo], fPQ[:nmo, :nmo]
                )
            B -= lib.einsum("qa,qa->", tmp, RRiQj[:nmo, i, nocc:nmo, i])
            if assume_gbc:
                tmp = lib.einsum(
                    "kc,kk->kc", RmPnQ[i, :nocc, i, nmo:], fPQ[:nocc, :nocc]
                )
                B -= 2 * lib.einsum("kc,kc->", tmp, RRiQj[:nocc, i, nmo:, i])
            else:
                tmp = lib.einsum("kc,kP->Pc", RmPnQ[i, :nocc, i, nmo:], fPQ[:nocc])
                B -= 2 * lib.einsum("Pc,Pc->", tmp, RRiQj[:, i, nmo:, i])
            if assume_gbc:
                tmp = lib.einsum(
                    "cb,ca->ab", RmPnQ[i, nmo:, i, nocc:nmo], fPQ[nmo:, nocc:nmo]
                )
                B -= 2 * lib.einsum("ab,ab->", tmp, RRiQj[nocc:nmo, i, nocc:nmo, i])
            else:
                tmp = lib.einsum(
                    "cb,cp->pb", RmPnQ[i, nmo:, i, nocc:nmo], fPQ[nmo:, :nmo]
                )
                B -= 2 * lib.einsum("pb,pb->", tmp, RRiQj[:nmo, i, nocc:nmo, i])
            B += 0.5 * tau[i, i, i, i]
            e += 2 * (-0.5 / zeta) * (-0.5 / zeta) * B
        return e

    e = add_B_contribution(zeta, nocc, frozen, Rbar_miPj, hPQ, RmPnQ_kPQ, RRiQj,
                           fPQ, tau, RmPnk_fPQ, RmPnQ)
    emp2_f12 += e
    del tau, RRiQj, RmPnQ_kPQ, RmPnk_fPQ
    del Rbar_miPj

    X_dict = {}
    Xtmp_dict = {}

    def get_X(o, l, RmPnQ):
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
            X -= lib.einsum(
                o + "p" + l + "q,piqj->" + xstring,
                RmPnQ[:, :nmo, :, :nmo],
                RmPnQ[:, :nmo, :, :nmo].transpose(1, 0, 3, 2).copy().conj(),
            )
            X -= lib.einsum(
                o + "z" + l + "c,zicj->" + xstring,
                RmPnQ[:, :nocc, :, nmo:],
                RmPnQ[:, :nocc, :, nmo:].transpose(1, 0, 3, 2).copy().conj(),
            )
            X -= lib.einsum(
                o + "c" + l + "z,cizj->" + xstring,
                RmPnQ[:, nmo:, :, :nocc],
                RmPnQ[:, :nocc, :, nmo:].transpose(3, 2, 1, 0).copy().conj(),
            )
            X_dict[ext_xstring] = X
        return X, xstring

    def get_Xtmp(m, n, RmPnQ):
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
            X, xstring = get_X(m, n, RmPnQ)
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
            X, xstring = get_X(n, m, RmPnQ)
            tmp += lib.einsum(
                m + n + n + m + "," + xstring + "->" + xtmpstring, tminj, X
            )
            Xtmp_dict[ext_xtmpstring] = tmp
        return tmp, xtmpstring

    def add_X_contribution(m, n, RmPnQ):
        # m, n = i, j or m, n = j, i
        xtmp, xtmpstring = get_Xtmp("k", n, RmPnQ)
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

        xtmp, xtmpstring = get_Xtmp(m, "k", RmPnQ)
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

    if not assume_gbc:
        emp2_f12 += add_X_contribution("i", "j", RmPnQ)
        emp2_f12 += add_X_contribution("j", "i", RmPnQ)
    else:
        X = Rbar_minj.copy()
        X -= lib.einsum(
            "mpnq,piqj->minj",
            RmPnQ[:, :nmo, :, :nmo],
            RmPnQ[:, :nmo, :, :nmo].transpose(1, 0, 3, 2).copy().conj(),
        )
        X -= lib.einsum(
            "mlnc,licj->minj",
            RmPnQ[:, :nocc, :, nmo:],
            RmPnQ[:, :nocc, :, nmo:].transpose(1, 0, 3, 2).copy().conj(),
        )
        X -= lib.einsum(
            "mcnl,cilj->minj",
            RmPnQ[:, nmo:, :, :nocc],
            RmPnQ[:, :nocc, :, nmo:].transpose(3, 2, 1, 0).copy().conj(),
        )
        fac_iijj = (-3.0 / (8 * zeta)) * (
            -2 * (-3.0 / (8 * zeta)) + (-1.0 / (8 * zeta))
        ) + (-1.0 / (8 * zeta)) * ((-3.0 / (8 * zeta)) - 2 * (-1.0 / (8 * zeta)))
        fac_ijji = (-1.0 / (8 * zeta)) * (
            -2 * (-3.0 / (8 * zeta)) + (-1.0 / (8 * zeta))
        ) + (-3.0 / (8 * zeta)) * ((-3.0 / (8 * zeta)) - 2 * (-1.0 / (8 * zeta)))
        for i in range(nocc - frozen):
            for j in range(nocc - frozen):
                if i == j:
                    emp2_f12 += (
                        fPQ[i + frozen, i + frozen]
                        * (-2)
                        * (-0.5 / zeta)
                        * (-0.5 / zeta)
                        * X[i, i, i, i]
                    )
                else:
                    emp2_f12 += (
                        (fPQ[i + frozen, i + frozen] + fPQ[j + frozen, j + frozen])
                        * fac_iijj
                        * X[i, i, j, j]
                    )
                    emp2_f12 += (
                        (fPQ[i + frozen, i + frozen] + fPQ[j + frozen, j + frozen])
                        * fac_ijji
                        * X[j, i, i, j]
                    )

    del RmPnQ

    cabs_singles = mp2f12.get_cabs_singles(mf.mo_energy, fPQ, nocc, nmo)
    print("return f12 extra (without CABS)", emp2_f12)
    return emp2_f12


class DF_aux:
    def __init__(self, auxmol, mp, zeta):
        self.auxmol = auxmol
        self.stdout = mp.stdout
        self.verbose = mp.verbose
        self.auxmol.set_f12_zeta(zeta)


def _init_mp_df_eris_direct(
    with_df, mol, coeff1, coeff2, max_memory, int_type_suff="", h5obj=None, log=None
):
    """Adapted from df mp2."""
    from pyscf import gto
    from pyscf.df.incore import fill_2c2e
    from pyscf.ao2mo.outcore import balance_partition

    if log is None:
        log = logger.new_logger(with_df)

    with_df.mol = mol
    auxmol = with_df.auxmol
    nbas = mol.nbas
    nao, n1 = coeff1.shape
    n2 = coeff2.shape[1]
    nao_pair = nao * (nao + 1) // 2
    naoaux = auxmol.nao_nr()

    dtype = coeff1.dtype
    assert dtype == numpy.float64
    dsize = 8

    mo = numpy.asarray(numpy.hstack((coeff1, coeff2)), order="F")
    ijslice = (0, n1, n1, n2 + n1)

    tspans = numpy.zeros((5, 2))
    tnames = ["j2c", "j3c", "xform", "save", "fit"]
    tick = (logger.process_clock(), logger.perf_counter())
    # precompute for fitting
    j2c = fill_2c2e(mol, auxmol, intor="int2c2e" + int_type_suff)
    try:
        m2c = scipy.linalg.cholesky(j2c, lower=True)
        tag = "cd"
    except scipy.linalg.LinAlgError:
        e, u = numpy.linalg.eigh(j2c)
        cond = abs(e).max() / abs(e).min()
        # check whether neg e entries can be deleted for sqrt later.
        # keep = abs(e) > THRESH_LINDEP
        keep = e > THRESH_LINDEP
        log.debug("cond(j2c) = %g", cond)
        log.debug("keep %d/%d cderi vectors", numpy.count_nonzero(keep), keep.size)
        e = e[keep]
        u = u[:, keep]
        m2c = lib.dot(u * e**-0.5, u.T.conj())
        tag = "eig"
    j2c = None
    naux = m2c.shape[1]
    tock = (logger.process_clock(), logger.perf_counter())
    tspans[0] += numpy.asarray(tock) - numpy.asarray(tick)

    mem_avail = max_memory - lib.current_memory()[0]
    incore = h5obj is None
    if incore:
        ovL = numpy.empty((n1 * n2, naoaux), dtype=dtype)
        mem_avail -= ovL.size * dsize / 1e6
    else:
        raise RuntimeError("not incore is not tested")
        ovL_shape = (n1 * n2, naux)
        ovL = h5obj.create_dataset(
            "ovL", ovL_shape, dtype=dtype, chunks=(1, *ovL_shape[1:])
        )
        h5tmp = lib.H5TmpFile()
        Lov0_shape = (naoaux, n1 * n2)
        Lov0 = h5tmp.create_dataset(
            "Lov0", Lov0_shape, dtype=dtype, chunks=(1, *Lov0_shape[1:])
        )

    # buffer
    mem_blk = nao_pair * 2 * dsize / 1e6
    aux_blksize = max(1, min(naoaux, int(numpy.floor(mem_avail * 0.7 / mem_blk))))
    auxshl_range = balance_partition(auxmol.ao_loc, aux_blksize)
    auxlen = max([x[2] for x in auxshl_range])
    log.info(
        "mem_avail = %.2f  mem_blk = %.2f  auxlen = %d", mem_avail, mem_blk, auxlen
    )
    buf0 = numpy.empty(auxlen * nao_pair, dtype=dtype)
    buf0T = numpy.empty(auxlen * nao_pair, dtype=dtype)

    # precompute for j3c
    comp = 1
    aosym = "s2ij"
    int3c = gto.moleintor.ascint3(mol._add_suffix("int3c2e" + int_type_suff))
    atm_f, bas_f, env_f = gto.mole.conc_env(
        mol._atm, mol._bas, mol._env, auxmol._atm, auxmol._bas, auxmol._env
    )
    ao_loc_f = gto.moleintor.make_loc(bas_f, int3c)
    cintopt = gto.moleintor.make_cintopt(atm_f, bas_f, env_f, int3c)

    def calc_j3c_ao(kshl0, kshl1):
        shls_slice = (0, nbas, 0, nbas, nbas + kshl0, nbas + kshl1)
        pqL = gto.moleintor.getints3c(
            int3c,
            atm_f,
            bas_f,
            env_f,
            shls_slice,
            comp,
            aosym,
            ao_loc_f,
            cintopt,
            out=buf0,
        )
        Lpq = lib.transpose(pqL, out=buf0T)
        pqL = None
        return Lpq

    # transform
    k1 = 0
    for auxshl_rg in auxshl_range:
        kshl0, kshl1, dk = auxshl_rg
        k0, k1 = k1, k1 + dk
        log.debug(
            "kshl = [%d:%d/%d]  [%d:%d/%d]", kshl0, kshl1, auxmol.nbas, k0, k1, naoaux
        )
        tick = (logger.process_clock(), logger.perf_counter())
        lpq = calc_j3c_ao(kshl0, kshl1)
        tock = (logger.process_clock(), logger.perf_counter())
        tspans[1] += numpy.asarray(tock) - numpy.asarray(tick)
        lov = _ao2mo.nr_e2(lpq, mo, ijslice, aosym="s2", out=buf0)
        tick = (logger.process_clock(), logger.perf_counter())
        tspans[2] += numpy.asarray(tick) - numpy.asarray(tock)
        if incore:
            ovl = lib.transpose(lov, out=buf0T)
            ovL[:, k0:k1] = ovl
            ovl = None
        else:
            Lov0[k0:k1] = lov
        lpq = lov = None
        tock = (logger.process_clock(), logger.perf_counter())
        tspans[3] += numpy.asarray(tock) - numpy.asarray(tick)
    buf0 = buf0T = None
    tick = (logger.process_clock(), logger.perf_counter())
    # fit
    if tag == "cd":
        drv = getattr(libmp, "trisolve_parallel_grp", None)
    if incore:
        if tag == "cd":
            if drv is None:
                # TODO: check:
                # I think here naux = naoaux. In general naux < naoaux I think.
                # m2c shape: naoaux naux
                # ovL shape: n1*n2  naoaux
                ovL = scipy.linalg.solve_triangular(
                    m2c, ovL.T, lower=True, overwrite_b=True, check_finite=False
                ).T
            else:
                assert m2c.flags.f_contiguous
                grpfac = 10
                drv(
                    m2c.ctypes.data_as(ctypes.c_void_p),
                    ovL.ctypes.data_as(ctypes.c_void_p),
                    ctypes.c_int(naux),
                    ctypes.c_int(n1 * n2),
                    ctypes.c_int(grpfac),
                )
        else:
            nvxao = n2 * naoaux
            nvx = n2 * naux
            mem_blk = nvx * dsize / 1e6
            occ_blksize = max(1, min(n1, int(numpy.floor(mem_avail * 0.5 / mem_blk))))
            buf = numpy.empty(occ_blksize * nvx, dtype=dtype)
            ovL = ovL.reshape(-1)
            for i0, i1 in lib.prange(0, n1, occ_blksize):
                n1i = i1 - i0
                out = numpy.ndarray((n1i * n2, naux), dtype=dtype, buffer=buf)
                lib.dot(
                    ovL[i0 * nvxao : i1 * nvxao].reshape(n1i * n2, naoaux), m2c, c=out
                )
                ovL[i0 * nvx : i1 * nvx] = out.reshape(-1)
            ovL = ovL[: n1 * nvx].reshape(n1 * n2, naux)
            buf = None
    else:
        nvxao = n2 * naoaux
        nvx = n2 * naux
        mem_blk = nvxao * dsize / 1e6
        occ_blksize = max(1, min(n1, int(numpy.floor(mem_avail * 0.4 / mem_blk))))
        for i0, i1 in lib.prange(0, n1, occ_blksize):
            n1i = i1 - i0
            ivL = numpy.asarray(Lov0[:, i0 * n2 : i1 * n2].T, order="C")
            if tag == "cd":
                if drv is None:
                    ivL = scipy.linalg.solve_triangular(
                        m2c, ivL.T, lower=True, overwrite_b=True, check_finite=False
                    ).T
                else:
                    assert m2c.flags.f_contiguous
                    grpfac = 10
                    drv(
                        m2c.ctypes.data_as(ctypes.c_void_p),
                        ivL.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_int(naux),
                        ctypes.c_int(n1i * n2),
                        ctypes.c_int(grpfac),
                    )
            else:
                ivL = lib.dot(ivL.reshape(n1i * n2, naoaux), m2c)
            ovL[i0 * n2 : i1 * n2] = ivL
        del h5tmp["Lov0"]
        h5tmp.close()
        Lov0 = None
    tock = (logger.process_clock(), logger.perf_counter())
    tspans[4] += numpy.asarray(tock) - numpy.asarray(tick)

    for tspan, tname in zip(tspans, tnames):
        log.debug(
            "ao2mo CPU time for %-10s  %9.2f sec  wall time %9.2f sec", tname, *tspan
        )
    log.info("")
    return ovL


if __name__ == "__main__":
    bas = "d"
    b = 1.8
    natoms = 7
    assume_gbc = True
    special_df = False
    import time

    mol = gto.Mole()
    mol.atom = ""
    for i in range(natoms):
        d = 1.5 * i
        mol.atom += f"C 0 0 {d};"
    print(mol.atom)
    mol.basis = "augccpv" + bas + "z"
    mol.build()
    df_auxmol = mol.copy()
    df_auxmol.basis = "augcc-pv" + bas + "z-jkfit"
    # df_auxmol.basis = df.aug_etb(mol, beta=b)
    # df_auxmol.basis = "ccpvqz"
    df_auxmol.build(False, False)

    mf = scf.RHF(mol).density_fit(auxbasis=df_auxmol.basis)
    # mf.max_cycle = 2
    t0 = time.time()
    print(mf.scf())
    print("HF time", time.time() - t0)

    t1 = time.time()
    mymp = mp2.MP2(mf)
    e = mymp.kernel()[0]
    t2 = time.time()
    print("MP2 time", t2 - t1)
    auxmol = mol.copy()
    auxmol.basis = {"C": "aug-cc-pv" + bas + "z-optri.0.nw"}
    # auxmol.basis = ('ccpvqz-fit', 'cc-pVQZ-F12-OptRI')
    # auxmol.basis = 'cc-pVQZ'
    auxmol.build(False, False)
    print("MP2", e)
    t1 = time.time()
    e_f12 = energy_f12(
        mf, mymp, auxmol, df_auxmol, 1, frozen=natoms, assume_gbc=assume_gbc
    )
    difft = time.time() - t1
    e += e_f12
    print("MP2-F12", e)
    print("e_tot", e + mf.e_tot)
    print("e f12", e_f12, "time taken", difft)
    print("diff to assume gbc", e_f12 - -0.2097133285439657)
    print("diff", e_f12 - -0.20962263319499966)
    print("diff after dm asarray change", e_f12 - -0.2096226331959131)
    print("diff after Gpi2 change", e_f12 - -0.2096226331958522)
