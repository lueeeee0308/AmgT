/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

#include "_hypre_struct_ls.h"
#include "_hypre_struct_mv.hpp"
#include "pfmg.h"

/*--------------------------------------------------------------------------
 * Macro to "change coordinates".  This routine is written as though
 * coarsening is being done in the z-direction.  This macro is used to
 * allow for coarsening to be done in the x- and y-directions also.
 *--------------------------------------------------------------------------*/

#define MapIndex(in_index, cdir, out_index)                     \
   hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 2);   \
   cdir = (cdir + 1) % 3;                                       \
   hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 0);   \
   cdir = (cdir + 1) % 3;                                       \
   hypre_IndexD(out_index, cdir) = hypre_IndexD(in_index, 1);   \
   cdir = (cdir + 1) % 3;

/*--------------------------------------------------------------------------
 *  Sets up new coarse grid operator stucture.
 *--------------------------------------------------------------------------*/

hypre_StructMatrix *
hypre_PFMG3CreateRAPOp( hypre_StructMatrix *R,
                        hypre_StructMatrix *A,
                        hypre_StructMatrix *P,
                        hypre_StructGrid   *coarse_grid,
                        HYPRE_Int           cdir        )
{
   HYPRE_UNUSED_VAR(R);
   HYPRE_UNUSED_VAR(P);

   hypre_StructMatrix    *RAP;

   hypre_Index           *RAP_stencil_shape;
   hypre_StructStencil   *RAP_stencil;
   HYPRE_Int              RAP_stencil_size;
   HYPRE_Int              RAP_stencil_dim;
   HYPRE_Int              RAP_num_ghost[] = {1, 1, 1, 1, 1, 1};

   hypre_StructStencil   *A_stencil;
   HYPRE_Int              A_stencil_size;

   hypre_Index            index_temp;
   HYPRE_Int              k, j, i;
   HYPRE_Int              stencil_rank;

   RAP_stencil_dim = 3;

   A_stencil = hypre_StructMatrixStencil(A);
   A_stencil_size = hypre_StructStencilSize(A_stencil);

   /*-----------------------------------------------------------------------
    * Define RAP_stencil
    *-----------------------------------------------------------------------*/

   stencil_rank = 0;

   /*-----------------------------------------------------------------------
    * non-symmetric case
    *-----------------------------------------------------------------------*/

   /*-----------------------------------------------------------------------
    * 7-point fine grid stencil produces 19 point RAP
    *
    * Store all 27 elements except for the corners.
    *
    * For symmetric A, only store the lower triangular part, where
    * lower triangular means the lower triangular part on the matrix
    * in the standard lexicographic ordering.
    *-----------------------------------------------------------------------*/
   if ( A_stencil_size == 7)
   {
      RAP_stencil_size = 19;
      if (hypre_StructMatrixSymmetric(A))
      {
         RAP_stencil_size = (RAP_stencil_size + 1) / 2;
      }
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index,  RAP_stencil_size, HYPRE_MEMORY_HOST);
      for (k = -1; k < 2; k++)
      {
         for (j = -1; j < 2; j++)
         {
            for (i = -1; i < 2; i++)
            {
               if ((i * j * k == 0) && (stencil_rank < RAP_stencil_size))
               {
                  hypre_SetIndex3(index_temp, i, j, k);
                  MapIndex(index_temp, cdir,
                           RAP_stencil_shape[stencil_rank]);
                  stencil_rank++;
               }
            }
         }
      }
   }

   /*-----------------------------------------------------------------------
    * 19 or 27 point fine grid stencil produces 27 point RAP
    *
    * Store all 27 elements
    *
    * For symmetric A, only store the lower triangular part, where
    * lower triangular means the lower triangular part on the matrix
    * in the standard lexicographic ordering.
    *-----------------------------------------------------------------------*/
   else
   {
      RAP_stencil_size = 27;
      if (hypre_StructMatrixSymmetric(A))
      {
         RAP_stencil_size = (RAP_stencil_size + 1) / 2;
      }
      RAP_stencil_shape = hypre_CTAlloc(hypre_Index,  RAP_stencil_size, HYPRE_MEMORY_HOST);
      for (k = -1; k < 2; k++)
      {
         for (j = -1; j < 2; j++)
         {
            for (i = -1; i < 2; i++)
            {
               if (stencil_rank < RAP_stencil_size)
               {
                  hypre_SetIndex3(index_temp, i, j, k);
                  MapIndex(index_temp, cdir,
                           RAP_stencil_shape[stencil_rank]);
                  stencil_rank++;
               }
            }
         }
      }
   }

   RAP_stencil = hypre_StructStencilCreate(RAP_stencil_dim, RAP_stencil_size,
                                           RAP_stencil_shape);
   RAP = hypre_StructMatrixCreate(hypre_StructMatrixComm(A),
                                  coarse_grid, RAP_stencil);

   hypre_StructStencilDestroy(RAP_stencil);

   /*-----------------------------------------------------------------------
    * Coarse operator in symmetric iff fine operator is
    *-----------------------------------------------------------------------*/
   hypre_StructMatrixSymmetric(RAP) = hypre_StructMatrixSymmetric(A);

   /*-----------------------------------------------------------------------
    * Set number of ghost points - one one each boundary
    *-----------------------------------------------------------------------*/
   hypre_StructMatrixSetNumGhost(RAP, RAP_num_ghost);

   return RAP;
}

/*--------------------------------------------------------------------------
 * Routines to build RAP. These routines are fairly general
 *  1) No assumptions about symmetry of A
 *  2) No assumption that R = transpose(P)
 *  3) 7, 19 or 27-point fine grid A
 *
 * I am, however, assuming that the c-to-c interpolation is the identity.
 *
 * I've written a two routines - hypre_PFMG3BuildRAPSym to build the lower
 * triangular part of RAP (including the diagonal) and
 * hypre_PFMG3BuildRAPNoSym to build the upper triangular part of RAP
 * (excluding the diagonal). So using symmetric storage, only the first
 * routine would be called. With full storage both would need to be called.
 *
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMG3BuildRAPSym( hypre_StructMatrix *A,
                        hypre_StructMatrix *P,
                        hypre_StructMatrix *R,
                        HYPRE_Int           cdir,
                        hypre_Index         cindex,
                        hypre_Index         cstride,
                        hypre_StructMatrix *RAP     )
{
   hypre_StructStencil  *fine_stencil;
   HYPRE_Int             fine_stencil_size;
   hypre_StructGrid     *fgrid;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   HYPRE_Int             fi, ci;
   HYPRE_Int             constant_coefficient;
   HYPRE_Int             constant_coefficient_A;

   fine_stencil = hypre_StructMatrixStencil(A);
   fine_stencil_size = hypre_StructStencilSize(fine_stencil);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(RAP);
   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);
   hypre_assert( constant_coefficient == 0 || constant_coefficient == 1 );
   hypre_assert( hypre_StructMatrixConstantCoefficient(R) == constant_coefficient );
   hypre_assert( hypre_StructMatrixConstantCoefficient(P) == constant_coefficient );
   if (constant_coefficient == 1 )
   {
      hypre_assert( constant_coefficient_A == 1 );
   }
   else
   {
      hypre_assert( constant_coefficient_A == 0 || constant_coefficient_A == 2 );
   }

   fi = 0;
   hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      /*--------------------------------------------------------------------
       * Switch statement to direct control to apropriate BoxLoop depending
       * on stencil size. Default is full 27-point.
       *-----------------------------------------------------------------*/

      switch (fine_stencil_size)
      {

         /*--------------------------------------------------------------
          * Loop for symmetric 7-point fine grid operator; produces a
          * symmetric 19-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (below-south, below-west,
          * below-center, below-east, below-north, center-south,
          * center-west, and center-center).
          *--------------------------------------------------------------*/

         case 7:

            if ( constant_coefficient == 1 )
            {
               hypre_PFMG3BuildRAPSym_onebox_FSS07_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            else
            {
               hypre_PFMG3BuildRAPSym_onebox_FSS07_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            break;

         /*--------------------------------------------------------------
          * Loop for symmetric 19-point fine grid operator; produces a
          * symmetric 27-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (below-southwest, below-south,
          * below-southeast, below-west, below-center, below-east,
          * below-northwest, below-north, below-northeast, center-southwest,
          * center-south, center-southeast, center-west, and center-center).
          *--------------------------------------------------------------*/

         case 19:

            if ( constant_coefficient == 1 )
            {
               hypre_PFMG3BuildRAPSym_onebox_FSS19_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            else
            {
               hypre_PFMG3BuildRAPSym_onebox_FSS19_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            break;

         /*--------------------------------------------------------------
          * Loop for symmetric 27-point fine grid operator; produces a
          * symmetric 27-point coarse grid operator. We calculate only the
          * lower triangular stencil entries: (below-southwest, below-south,
          * below-southeast, below-west, below-center, below-east,
          * below-northwest, below-north, below-northeast, center-southwest,
          * center-south, center-southeast, center-west, and center-center).
          *--------------------------------------------------------------*/

         default:

            if ( constant_coefficient == 1 )
            {
               hypre_PFMG3BuildRAPSym_onebox_FSS27_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            else
            {
               hypre_PFMG3BuildRAPSym_onebox_FSS27_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            break;

      } /* end switch statement */

   } /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPSym, for one box, one value of fine_stencil_size
   (7) and one value of constant_coefficient (0).  Within this function
   there is a test on constant_coefficient_A as well.  */
HYPRE_Int
hypre_PFMG3BuildRAPSym_onebox_FSS07_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient_A;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;
   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac;
   HYPRE_Real           *a_bc;
   HYPRE_Real            a_cs_offd, a_cs_offdm1, a_cs_offdp1;
   HYPRE_Real            a_cn_offdm1;
   HYPRE_Real            a_cw_offd, a_cw_offdm1, a_cw_offdp1;
   HYPRE_Real            a_ce_offdm1;
   HYPRE_Real            a_ac_offd, a_ac_offdm1;
   HYPRE_Real            a_bc_offd, a_bc_offdm1, a_bc_offdp1;
   HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_csw, *rap_cse;
   HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             zOffsetA_diag;
   HYPRE_Int             zOffsetA_offd;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   HYPRE_Int pbOffset = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET
   HYPRE_Int rbOffset = hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);


   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      zOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      zOffsetA_diag = hypre_BoxOffsetDistance(A_dbox, index);
      zOffsetA_offd = 0;
   }

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   /*--------------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 7-point fine grid operator; produces a
    * symmetric 19-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (below-south, below-west,
    * below-center, below-east, below-north, center-south,
    * center-west, and center-center).
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
#define DEVICE_VAR is_device_ptr(rap_bs,rb,a_cs,pa,rap_bw,a_cw,rap_bc,a_bc,a_cc,rap_be,a_ce,rap_bn,a_cn,rap_cs,pb,ra,rap_cw,rap_csw,rap_cse,rap_cc,a_ac)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc)
      {
         HYPRE_Int iAm1 = iA - zOffsetA;
         HYPRE_Int iAp1 = iA + zOffsetA;

         HYPRE_Int iP1 = iP - zOffsetP - yOffsetP;
         rap_bs[iAc] = rb[iR - rbOffset] * a_cs[iAm1] * pa[iP1];

         iP1 = iP - zOffsetP - xOffsetP;
         rap_bw[iAc] = rb[iR - rbOffset] * a_cw[iAm1] * pa[iP1];

         iP1 = iP - zOffsetP;
         rap_bc[iAc] =          a_bc[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                                +          rb[iR - rbOffset] * a_bc[iAm1];

         iP1 = iP - zOffsetP + xOffsetP;
         rap_be[iAc] = rb[iR - rbOffset] * a_ce[iAm1] * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP;
         rap_bn[iAc] = rb[iR - rbOffset] * a_cn[iAm1] * pa[iP1];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] =          a_cs[iA]
                                +          rb[iR - rbOffset] * a_cs[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cs[iAp1] * pa[iP1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw[iA]
                                +          rb[iR - rbOffset] * a_cw[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw[iAp1] * pa[iP1];

         rap_csw[iAc] = 0.0;

         rap_cse[iAc] = 0.0;

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_ac[iAm1]
                                +          ra[iR] * a_bc[iAp1]
                                +                   a_bc[iA]   * pb[iP - pbOffset]
                                +                   a_ac[iA]   * pa[iP];

      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      iA_offd = 0;
      iA_offdm1 = iA_offd - zOffsetA_offd;
      iA_offdp1 = iA_offd + zOffsetA_offd;
      a_cs_offd = a_cs[iA_offd];
      a_cs_offdm1 = a_cs[iA_offdm1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cw_offd = a_cw[iA_offd];
      a_cw_offdm1 = a_cw[iA_offdm1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_bc_offd = a_bc[iA_offd];
      a_bc_offdm1 = a_bc[iA_offdm1];
      a_bc_offdp1 = a_bc[iA_offdp1];
      a_ac_offd = a_ac[iA_offd];
      a_ac_offdm1 = a_ac[iA_offdm1];

#define DEVICE_VAR is_device_ptr(rap_bs,rb,pa,rap_bw,rap_bc,a_cc,rap_be,rap_bn,rap_cs,pb,ra,rap_cw,rap_csw,rap_cse,rap_cc)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAm1 = iA - zOffsetA_diag;
         HYPRE_Int iAp1 = iA + zOffsetA_diag;

         HYPRE_Int iP1 = iP - zOffsetP - yOffsetP;
         rap_bs[iAc] = rb[iR - rbOffset] * a_cs_offdm1 * pa[iP1];

         iP1 = iP - zOffsetP - xOffsetP;
         rap_bw[iAc] = rb[iR - rbOffset] * a_cw_offdm1 * pa[iP1];

         iP1 = iP - zOffsetP;
         rap_bc[iAc] =          a_bc_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                                +          rb[iR - rbOffset] * a_bc_offdm1;

         iP1 = iP - zOffsetP + xOffsetP;
         rap_be[iAc] = rb[iR - rbOffset] * a_ce_offdm1 * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP;
         rap_bn[iAc] = rb[iR - rbOffset] * a_cn_offdm1 * pa[iP1];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] =          a_cs_offd
                                +          rb[iR - rbOffset] * a_cs_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cs_offdp1 * pa[iP1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw_offd
                                +          rb[iR - rbOffset] * a_cw_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw_offdp1 * pa[iP1];

         rap_csw[iAc] = 0.0;

         rap_cse[iAc] = 0.0;

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_ac_offdm1
                                +          ra[iR] * a_bc_offdp1
                                +                   a_bc_offd   * pb[iP - pbOffset]
                                +                   a_ac_offd   * pa[iP];

      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPSym, for one box, one value of fine_stencil_size
   (7) and one value of constant_coefficient (1). */
HYPRE_Int
hypre_PFMG3BuildRAPSym_onebox_FSS07_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           fstart;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac;
   HYPRE_Real           *a_bc;
   HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_csw, *rap_cse;
   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetA = 0;
   zOffsetP = 0;

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = 0;

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*--------------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 7-point fine grid operator; produces a
    * symmetric 19-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (below-south, below-west,
    * below-center, below-east, below-north, center-south,
    * center-west, and center-center).
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - zOffsetA;
   iAp1 = iA + zOffsetA;

   iP1 = iP - zOffsetP - yOffsetP;
   rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1];

   iP1 = iP - zOffsetP - xOffsetP;
   rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1];

   iP1 = iP - zOffsetP;
   rap_bc[iAc] =          a_bc[iA]   * pa[iP1]
                          +          rb[iR] * a_cc[iAm1] * pa[iP1]
                          +          rb[iR] * a_bc[iAm1];

   iP1 = iP - zOffsetP + xOffsetP;
   rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1];

   iP1 = iP - zOffsetP + yOffsetP;
   rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1];

   iP1 = iP - yOffsetP;
   rap_cs[iAc] =          a_cs[iA]
                          +          rb[iR] * a_cs[iAm1] * pb[iP1]
                          +          ra[iR] * a_cs[iAp1] * pa[iP1];

   iP1 = iP - xOffsetP;
   rap_cw[iAc] =          a_cw[iA]
                          +          rb[iR] * a_cw[iAm1] * pb[iP1]
                          +          ra[iR] * a_cw[iAp1] * pa[iP1];

   rap_csw[iAc] = 0.0;

   rap_cse[iAc] = 0.0;

   rap_cc[iAc] =          a_cc[iA]
                          +          rb[iR] * a_cc[iAm1] * pb[iP]
                          +          ra[iR] * a_cc[iAp1] * pa[iP]
                          +          rb[iR] * a_ac[iAm1]
                          +          ra[iR] * a_bc[iAp1]
                          +                   a_bc[iA]   * pb[iP]
                          +                   a_ac[iA]   * pa[iP];


   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPSym, for one box, one value of fine_stencil_size
   (19) and one value of constant_coefficient (0).  Within this functions
   there is a test on constant_coefficient_A as well.  */
HYPRE_Int
hypre_PFMG3BuildRAPSym_onebox_FSS19_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient_A;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw, *a_as;
   HYPRE_Real           *a_bc, *a_bw, *a_be, *a_bs, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   HYPRE_Real            a_cs_offd, a_cs_offdm1, a_cs_offdp1;
   HYPRE_Real            a_csw_offd, a_csw_offdm1, a_csw_offdp1;
   HYPRE_Real            a_cse_offd, a_cse_offdm1, a_cse_offdp1;
   HYPRE_Real            a_cn_offdm1, a_cne_offdm1, a_cnw_offdm1;
   HYPRE_Real            a_cw_offd, a_cw_offdm1, a_cw_offdp1;
   HYPRE_Real            a_ce_offdm1;
   HYPRE_Real            a_ac_offd, a_ac_offdm1;
   HYPRE_Real            a_aw_offd, a_aw_offdm1;
   HYPRE_Real            a_as_offd, a_as_offdm1;
   HYPRE_Real            a_bc_offd, a_bc_offdm1, a_bc_offdp1;
   HYPRE_Real            a_be_offd, a_be_offdm1;
   HYPRE_Real            a_bn_offd, a_bn_offdm1;
   HYPRE_Real            a_bw_offd, a_bw_offdm1, a_bw_offdp1;
   HYPRE_Real            a_bs_offd, a_bs_offdm1, a_bs_offdp1;

   HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_csw, *rap_cse;
   HYPRE_Real           *rap_bsw, *rap_bse, *rap_bnw, *rap_bne;

   HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             zOffsetA_diag;
   HYPRE_Int             zOffsetA_offd;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   HYPRE_Int pbOffset = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET
   HYPRE_Int rbOffset = hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);


   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point coarse grid operator:
    *
    * A 27-point coarse grid operator is produced when the fine grid
    * stencil is 19 or 27 point.
    *
    * We build only the lower triangular part.
    *
    * rap_csw is pointer for southwest coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bsw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   zOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      zOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      zOffsetA_diag = hypre_BoxOffsetDistance(A_dbox, index);
      zOffsetA_offd = 0;
   }

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   /*--------------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 19-point fine grid operator; produces a
    * symmetric 27-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (below-southwest, below-south,
    * below-southeast, below-west, below-center, below-east,
    * below-northwest, below-north, below-northeast, center-southwest,
    * center-south, center-southeast, center-west, and center-center).
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
#define DEVICE_VAR is_device_ptr(rap_bsw,rb,a_csw,pa,rap_bs,a_cs,a_bs,rap_bse,a_cse,rap_bw,a_cw,a_bw,rap_bc,a_bc,a_cc,rap_be,a_ce,a_be,rap_bnw,a_cnw,rap_bn,a_cn,a_bn,rap_bne,a_cne,rap_csw,pb,ra,rap_cs,a_as,rap_cse,rap_cw,a_aw,rap_cc,a_ac)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAm1 = iA - zOffsetA;
         HYPRE_Int iAp1 = iA + zOffsetA;

         HYPRE_Int iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
         rap_bsw[iAc] = rb[iR - rbOffset] * a_csw[iAm1] * pa[iP1];

         iP1 = iP - zOffsetP - yOffsetP;
         rap_bs[iAc] = rb[iR - rbOffset] * a_cs[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_bs[iAm1]
                       +                   a_bs[iA]   * pa[iP1];

         iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
         rap_bse[iAc] = rb[iR - rbOffset] * a_cse[iAm1] * pa[iP1];

         iP1 = iP - zOffsetP - xOffsetP;
         rap_bw[iAc] = rb[iR - rbOffset] * a_cw[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_bw[iAm1]
                       +                   a_bw[iA]   * pa[iP1];

         iP1 = iP - zOffsetP;
         rap_bc[iAc] =          a_bc[iA] * pa[iP1]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                                +          rb[iR - rbOffset] * a_bc[iAm1];

         iP1 = iP - zOffsetP + xOffsetP;
         rap_be[iAc] = rb[iR - rbOffset] * a_ce[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_be[iAm1]
                       +                   a_be[iA]   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
         rap_bnw[iAc] = rb[iR - rbOffset] * a_cnw[iAm1] * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP;
         rap_bn[iAc] = rb[iR - rbOffset] * a_cn[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_bn[iAm1]
                       +                   a_bn[iA]   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
         rap_bne[iAc] = rb[iR - rbOffset] * a_cne[iAm1] * pa[iP1];

         iP1 = iP - yOffsetP - xOffsetP;
         rap_csw[iAc] =         a_csw[iA]
                                +          rb[iR - rbOffset] * a_csw[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_csw[iAp1] * pa[iP1];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] =          a_cs[iA]
                                +          rb[iR - rbOffset] * a_cs[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cs[iAp1] * pa[iP1]
                                +                   a_bs[iA]   * pb[iP1 - pbOffset]
                                +                   a_as[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_as[iAm1]
                                +          ra[iR] * a_bs[iAp1];

         iP1 = iP - yOffsetP + xOffsetP;
         rap_cse[iAc] =          a_cse[iA]
                                 +          rb[iR - rbOffset] * a_cse[iAm1] * pb[iP1 - pbOffset]
                                 +          ra[iR] * a_cse[iAp1] * pa[iP1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw[iA]
                                +          rb[iR - rbOffset] * a_cw[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw[iAp1] * pa[iP1]
                                +                   a_bw[iA]   * pb[iP1 - pbOffset]
                                +                   a_aw[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_aw[iAm1]
                                +          ra[iR] * a_bw[iAp1];

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_ac[iAm1]
                                +          ra[iR] * a_bc[iAp1]
                                +                   a_bc[iA]   * pb[iP - pbOffset]
                                +                   a_ac[iA]   * pa[iP];

      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      iA_offd = 0;
      iA_offdm1 = iA_offd - zOffsetA_offd;
      iA_offdp1 = iA_offd + zOffsetA_offd;
      a_cs_offd = a_cs[iA_offd];
      a_cs_offdm1 = a_cs[iA_offdm1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cw_offd = a_cw[iA_offd];
      a_cw_offdm1 = a_cw[iA_offdm1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_csw_offd = a_csw[iA_offd];
      a_csw_offdm1 = a_csw[iA_offdm1];
      a_csw_offdp1 = a_csw[iA_offdp1];
      a_cse_offd = a_cse[iA_offd];
      a_cse_offdm1 = a_cse[iA_offdm1];
      a_cse_offdp1 = a_cse[iA_offdp1];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cne_offdm1 = a_cne[iA_offdm1];
      a_cnw_offdm1 = a_cnw[iA_offdm1];
      a_ac_offd = a_ac[iA_offd];
      a_ac_offdm1 = a_ac[iA_offdm1];
      a_aw_offd = a_aw[iA_offd];
      a_aw_offdm1 = a_aw[iA_offdm1];
      a_as_offd = a_as[iA_offd];
      a_as_offdm1 = a_as[iA_offdm1];
      a_bc_offd = a_bc[iA_offd];
      a_bc_offdm1 = a_bc[iA_offdm1];
      a_bc_offdp1 = a_bc[iA_offdp1];
      a_be_offd = a_be[iA_offd];
      a_be_offdm1 = a_be[iA_offdm1];
      a_bn_offd = a_bn[iA_offd];
      a_bn_offdm1 = a_bn[iA_offdm1];
      a_bw_offd = a_bw[iA_offd];
      a_bw_offdm1 = a_bw[iA_offdm1];
      a_bw_offdp1 = a_bw[iA_offdp1];
      a_bs_offd = a_bs[iA_offd];
      a_bs_offdm1 = a_bs[iA_offdm1];
      a_bs_offdp1 = a_bs[iA_offdp1];

#define DEVICE_VAR is_device_ptr(rap_bsw,rb,pa,rap_bs,rap_bse,rap_bw,rap_bc,a_cc,rap_be,rap_bnw,rap_bn,rap_bne,rap_csw,pb,ra,rap_cs,rap_cse,rap_cw,rap_cc)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAm1 = iA - zOffsetA_diag;
         HYPRE_Int iAp1 = iA + zOffsetA_diag;

         HYPRE_Int iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
         rap_bsw[iAc] = rb[iR - rbOffset] * a_csw_offdm1 * pa[iP1];

         iP1 = iP - zOffsetP - yOffsetP;
         rap_bs[iAc] = rb[iR - rbOffset] * a_cs_offdm1 * pa[iP1]
                       +          rb[iR - rbOffset] * a_bs_offdm1
                       +                   a_bs_offd   * pa[iP1];

         iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
         rap_bse[iAc] = rb[iR - rbOffset] * a_cse_offdm1 * pa[iP1];

         iP1 = iP - zOffsetP - xOffsetP;
         rap_bw[iAc] = rb[iR - rbOffset] * a_cw_offdm1 * pa[iP1]
                       +          rb[iR - rbOffset] * a_bw_offdm1
                       +                   a_bw_offd   * pa[iP1];

         iP1 = iP - zOffsetP;
         rap_bc[iAc] =          a_bc_offd * pa[iP1]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                                +          rb[iR - rbOffset] * a_bc_offdm1;

         iP1 = iP - zOffsetP + xOffsetP;
         rap_be[iAc] = rb[iR - rbOffset] * a_ce_offdm1 * pa[iP1]
                       +          rb[iR - rbOffset] * a_be_offdm1
                       +                   a_be_offd   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
         rap_bnw[iAc] = rb[iR - rbOffset] * a_cnw_offdm1 * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP;
         rap_bn[iAc] = rb[iR - rbOffset] * a_cn_offdm1 * pa[iP1]
                       +          rb[iR - rbOffset] * a_bn_offdm1
                       +                   a_bn_offd   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
         rap_bne[iAc] = rb[iR - rbOffset] * a_cne_offdm1 * pa[iP1];

         iP1 = iP - yOffsetP - xOffsetP;
         rap_csw[iAc] =         a_csw_offd
                                +          rb[iR - rbOffset] * a_csw_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_csw_offdp1 * pa[iP1 - pbOffset];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] =          a_cs_offd
                                +          rb[iR - rbOffset] * a_cs_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cs_offdp1 * pa[iP1]
                                +                   a_bs_offd   * pb[iP1 - pbOffset]
                                +                   a_as_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_as_offdm1
                                +          ra[iR] * a_bs_offdp1;

         iP1 = iP - yOffsetP + xOffsetP;
         rap_cse[iAc] =          a_cse_offd
                                 +          rb[iR - rbOffset] * a_cse_offdm1 * pb[iP1 - pbOffset]
                                 +          ra[iR] * a_cse_offdp1 * pa[iP1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw_offd
                                +          rb[iR - rbOffset] * a_cw_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw_offdp1 * pa[iP1]
                                +                   a_bw_offd   * pb[iP1 - pbOffset]
                                +                   a_aw_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_aw_offdm1
                                +          ra[iR] * a_bw_offdp1;

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_ac_offdm1
                                +          ra[iR] * a_bc_offdp1
                                +                   a_bc_offd   * pb[iP - pbOffset]
                                +                   a_ac_offd   * pa[iP];

      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPSym, for one box, one value of fine_stencil_size
   (19) and one value of constant_coefficient (1).  */
HYPRE_Int
hypre_PFMG3BuildRAPSym_onebox_FSS19_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           fstart;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw, *a_as;
   HYPRE_Real           *a_bc, *a_bw, *a_be, *a_bs, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;

   HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_csw, *rap_cse;
   HYPRE_Real           *rap_bsw, *rap_bse, *rap_bnw, *rap_bne;

   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point coarse grid operator:
    *
    * A 27-point coarse grid operator is produced when the fine grid
    * stencil is 19 or 27 point.
    *
    * We build only the lower triangular part.
    *
    * rap_csw is pointer for southwest coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bsw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetA = 0;
   zOffsetP = 0;

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = 0;

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*--------------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 19-point fine grid operator; produces a
    * symmetric 27-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (below-southwest, below-south,
    * below-southeast, below-west, below-center, below-east,
    * below-northwest, below-north, below-northeast, center-southwest,
    * center-south, center-southeast, center-west, and center-center).
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - zOffsetA;
   iAp1 = iA + zOffsetA;

   iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
   rap_bsw[iAc] = rb[iR] * a_csw[iAm1] * pa[iP1];

   iP1 = iP - zOffsetP - yOffsetP;
   rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                 +          rb[iR] * a_bs[iAm1]
                 +                   a_bs[iA]   * pa[iP1];

   iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
   rap_bse[iAc] = rb[iR] * a_cse[iAm1] * pa[iP1];

   iP1 = iP - zOffsetP - xOffsetP;
   rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                 +          rb[iR] * a_bw[iAm1]
                 +                   a_bw[iA]   * pa[iP1];

   iP1 = iP - zOffsetP;
   rap_bc[iAc] =          a_bc[iA] * pa[iP1]
                          +          rb[iR] * a_cc[iAm1] * pa[iP1]
                          +          rb[iR] * a_bc[iAm1];

   iP1 = iP - zOffsetP + xOffsetP;
   rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                 +          rb[iR] * a_be[iAm1]
                 +                   a_be[iA]   * pa[iP1];

   iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
   rap_bnw[iAc] = rb[iR] * a_cnw[iAm1] * pa[iP1];

   iP1 = iP - zOffsetP + yOffsetP;
   rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                 +          rb[iR] * a_bn[iAm1]
                 +                   a_bn[iA]   * pa[iP1];

   iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
   rap_bne[iAc] = rb[iR] * a_cne[iAm1] * pa[iP1];

   iP1 = iP - yOffsetP - xOffsetP;
   rap_csw[iAc] =         a_csw[iA]
                          +          rb[iR] * a_csw[iAm1] * pb[iP1]
                          +          ra[iR] * a_csw[iAp1] * pa[iP1];

   iP1 = iP - yOffsetP;
   rap_cs[iAc] =          a_cs[iA]
                          +          rb[iR] * a_cs[iAm1] * pb[iP1]
                          +          ra[iR] * a_cs[iAp1] * pa[iP1]
                          +                   a_bs[iA]   * pb[iP1]
                          +                   a_as[iA]   * pa[iP1]
                          +          rb[iR] * a_as[iAm1]
                          +          ra[iR] * a_bs[iAp1];

   iP1 = iP - yOffsetP + xOffsetP;
   rap_cse[iAc] =          a_cse[iA]
                           +          rb[iR] * a_cse[iAm1] * pb[iP1]
                           +          ra[iR] * a_cse[iAp1] * pa[iP1];

   iP1 = iP - xOffsetP;
   rap_cw[iAc] =          a_cw[iA]
                          +          rb[iR] * a_cw[iAm1] * pb[iP1]
                          +          ra[iR] * a_cw[iAp1] * pa[iP1]
                          +                   a_bw[iA]   * pb[iP1]
                          +                   a_aw[iA]   * pa[iP1]
                          +          rb[iR] * a_aw[iAm1]
                          +          ra[iR] * a_bw[iAp1];

   rap_cc[iAc] =          a_cc[iA]
                          +          rb[iR] * a_cc[iAm1] * pb[iP]
                          +          ra[iR] * a_cc[iAp1] * pa[iP]
                          +          rb[iR] * a_ac[iAm1]
                          +          ra[iR] * a_bc[iAp1]
                          +                   a_bc[iA]   * pb[iP]
                          +                   a_ac[iA]   * pa[iP];

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPSym, for one box, one value of fine_stencil_size
   (27) and one value of constant_coefficient (0).  Within this functions
   there is a test on constant_coefficient_A as well.  */
HYPRE_Int
hypre_PFMG3BuildRAPSym_onebox_FSS27_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient_A;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw, *a_as;
   HYPRE_Real           *a_bc, *a_bw, *a_be, *a_bs, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   HYPRE_Real           *a_asw, *a_ase;
   HYPRE_Real           *a_bsw, *a_bse, *a_bnw, *a_bne;
   HYPRE_Real            a_cs_offd, a_cs_offdm1, a_cs_offdp1;
   HYPRE_Real            a_csw_offd, a_csw_offdm1, a_csw_offdp1;
   HYPRE_Real            a_cse_offd, a_cse_offdm1, a_cse_offdp1;
   HYPRE_Real            a_cn_offdm1, a_cne_offdm1, a_cnw_offdm1;
   HYPRE_Real            a_cw_offd, a_cw_offdm1, a_cw_offdp1;
   HYPRE_Real            a_ce_offdm1;
   HYPRE_Real            a_ac_offd, a_ac_offdm1;
   HYPRE_Real            a_aw_offd, a_aw_offdm1;
   HYPRE_Real            a_as_offd, a_as_offdm1;
   HYPRE_Real            a_asw_offd, a_asw_offdm1;
   HYPRE_Real            a_ase_offd, a_ase_offdm1;
   HYPRE_Real            a_bc_offd, a_bc_offdm1, a_bc_offdp1;
   HYPRE_Real            a_be_offd, a_be_offdm1;
   HYPRE_Real            a_bn_offd, a_bn_offdm1;
   HYPRE_Real            a_bw_offd, a_bw_offdm1, a_bw_offdp1;
   HYPRE_Real            a_bs_offd, a_bs_offdm1, a_bs_offdp1;
   HYPRE_Real            a_bsw_offd, a_bsw_offdm1, a_bsw_offdp1;
   HYPRE_Real            a_bse_offd, a_bse_offdm1, a_bse_offdp1;
   HYPRE_Real            a_bnw_offd, a_bnw_offdm1;
   HYPRE_Real            a_bne_offd, a_bne_offdm1;

   HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_csw, *rap_cse;
   HYPRE_Real           *rap_bsw, *rap_bse, *rap_bnw, *rap_bne;

   HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;


   HYPRE_Int             zOffsetA;
   HYPRE_Int             zOffsetA_diag;
   HYPRE_Int             zOffsetA_offd;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET:
   HYPRE_Int pbOffset = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET:
   HYPRE_Int rbOffset = hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point fine grid operator:
    *
    * a_asw is pointer for southwest coefficient in plane above
    * a_ase is pointer for southeast coefficient in plane above
    * a_anw is pointer for northwest coefficient in plane above
    * a_ane is pointer for northeast coefficient in plane above
    * a_bsw is pointer for southwest coefficient in plane below
    * a_bse is pointer for southeast coefficient in plane below
    * a_bnw is pointer for northwest coefficient in plane below
    * a_bne is pointer for northeast coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_asw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_ase = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, -1);
   MapIndex(index_temp, cdir, index);
   a_bsw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, -1);
   MapIndex(index_temp, cdir, index);
   a_bse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point coarse grid operator:
    *
    * A 27-point coarse grid operator is produced when the fine grid
    * stencil is 19 or 27 point.
    *
    * We build only the lower triangular part.
    *
    * rap_csw is pointer for southwest coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bsw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      zOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      zOffsetA_diag = hypre_BoxOffsetDistance(A_dbox, index);
      zOffsetA_offd = 0;
   }

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   /*--------------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 27-point fine grid operator; produces a
    * symmetric 27-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (below-southwest, below-south,
    * below-southeast, below-west, below-center, below-east,
    * below-northwest, below-north, below-northeast, center-southwest,
    * center-south, center-southeast, center-west, and center-center).
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
#define DEVICE_VAR is_device_ptr(rap_bsw,rb,a_csw,pa,a_bsw,rap_bs,a_cs,a_bs,rap_bse,a_cse,a_bse,rap_bw,a_cw,a_bw,rap_bc,a_bc,a_cc,rap_be,a_ce,a_be,rap_bnw,a_cnw,a_bnw,rap_bn,a_cn,a_bn,rap_bne,a_cne,a_bne,rap_csw,pb,ra,a_asw,rap_cs,a_as,rap_cse,a_ase,rap_cw,a_aw,rap_cc,a_ac)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAm1 = iA - zOffsetA;
         HYPRE_Int iAp1 = iA + zOffsetA;

         HYPRE_Int iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
         rap_bsw[iAc] = rb[iR - rbOffset] * a_csw[iAm1] * pa[iP1]
                        +           rb[iR - rbOffset] * a_bsw[iAm1]
                        +                    a_bsw[iA]   * pa[iP1];

         iP1 = iP - zOffsetP - yOffsetP;
         rap_bs[iAc] = rb[iR - rbOffset] * a_cs[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_bs[iAm1]
                       +                   a_bs[iA]   * pa[iP1];

         iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
         rap_bse[iAc] = rb[iR - rbOffset] * a_cse[iAm1] * pa[iP1]
                        +           rb[iR - rbOffset] * a_bse[iAm1]
                        +                    a_bse[iA]   * pa[iP1];

         iP1 = iP - zOffsetP - xOffsetP;
         rap_bw[iAc] = rb[iR - rbOffset] * a_cw[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_bw[iAm1]
                       +                   a_bw[iA]   * pa[iP1];

         iP1 = iP - zOffsetP;
         rap_bc[iAc] =          a_bc[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                                +          rb[iR - rbOffset] * a_bc[iAm1];

         iP1 = iP - zOffsetP + xOffsetP;
         rap_be[iAc] = rb[iR - rbOffset] * a_ce[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_be[iAm1]
                       +                   a_be[iA]   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
         rap_bnw[iAc] = rb[iR - rbOffset] * a_cnw[iAm1] * pa[iP1]
                        +           rb[iR - rbOffset] * a_bnw[iAm1]
                        +                    a_bnw[iA]   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP;
         rap_bn[iAc] = rb[iR - rbOffset] * a_cn[iAm1] * pa[iP1]
                       +          rb[iR - rbOffset] * a_bn[iAm1]
                       +                   a_bn[iA]   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
         rap_bne[iAc] = rb[iR - rbOffset] * a_cne[iAm1] * pa[iP1]
                        +           rb[iR - rbOffset] * a_bne[iAm1]
                        +                    a_bne[iA]   * pa[iP1];

         iP1 = iP - yOffsetP - xOffsetP;
         rap_csw[iAc] =          a_csw[iA]
                                 +          rb[iR - rbOffset] * a_csw[iAm1] * pb[iP1 - pbOffset]
                                 +          ra[iR] * a_csw[iAp1] * pa[iP1]
                                 +                   a_bsw[iA]   * pb[iP1 - pbOffset]
                                 +                   a_asw[iA]   * pa[iP1]
                                 +          rb[iR - rbOffset] * a_asw[iAm1]
                                 +          ra[iR] * a_bsw[iAp1];

         iP1 = iP - yOffsetP;
         rap_cs[iAc] =          a_cs[iA]
                                +          rb[iR - rbOffset] * a_cs[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cs[iAp1] * pa[iP1]
                                +                   a_bs[iA]   * pb[iP1 - pbOffset]
                                +                   a_as[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_as[iAm1]
                                +          ra[iR] * a_bs[iAp1];

         iP1 = iP - yOffsetP + xOffsetP;
         rap_cse[iAc] =          a_cse[iA]
                                 +          rb[iR - rbOffset] * a_cse[iAm1] * pb[iP1 - pbOffset]
                                 +          ra[iR] * a_cse[iAp1] * pa[iP1]
                                 +                   a_bse[iA]   * pb[iP1 - pbOffset]
                                 +                   a_ase[iA]   * pa[iP1]
                                 +          rb[iR - rbOffset] * a_ase[iAm1]
                                 +          ra[iR] * a_bse[iAp1];

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw[iA]
                                +          rb[iR - rbOffset] * a_cw[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw[iAp1] * pa[iP1]
                                +                   a_bw[iA]   * pb[iP1 - pbOffset]
                                +                   a_aw[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_aw[iAm1]
                                +          ra[iR] * a_bw[iAp1];

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_ac[iAm1]
                                +          ra[iR] * a_bc[iAp1]
                                +                   a_bc[iA]   * pb[iP - pbOffset]
                                +                   a_ac[iA]   * pa[iP];
      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      iA_offd = 0;
      iA_offdm1 = iA_offd - zOffsetA_offd;
      iA_offdp1 = iA_offd + zOffsetA_offd;
      a_cs_offd = a_cs[iA_offd];
      a_cs_offdm1 = a_cs[iA_offdm1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cse_offd = a_cse[iA_offd];
      a_cse_offdm1 = a_cse[iA_offdm1];
      a_cse_offdp1 = a_cse[iA_offdp1];
      a_csw_offd = a_csw[iA_offd];
      a_csw_offdm1 = a_csw[iA_offdm1];
      a_csw_offdp1 = a_csw[iA_offdp1];
      a_cw_offd = a_cw[iA_offd];
      a_cw_offdm1 = a_cw[iA_offdm1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cne_offdm1 = a_cne[iA_offdm1];
      a_cnw_offdm1 = a_cnw[iA_offdm1];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_ac_offd = a_ac[iA_offd];
      a_ac_offdm1 = a_ac[iA_offdm1];
      a_as_offd = a_as[iA_offd];
      a_as_offdm1 = a_as[iA_offdm1];
      a_aw_offd = a_aw[iA_offd];
      a_aw_offdm1 = a_aw[iA_offdm1];
      a_asw_offd = a_asw[iA_offd];
      a_asw_offdm1 = a_asw[iA_offdm1];
      a_ase_offd = a_ase[iA_offd];
      a_ase_offdm1 = a_ase[iA_offdm1];
      a_bc_offd = a_bc[iA_offd];
      a_bc_offdm1 = a_bc[iA_offdm1];
      a_bc_offdp1 = a_bc[iA_offdp1];
      a_bs_offd = a_bs[iA_offd];
      a_bs_offdm1 = a_bs[iA_offdm1];
      a_bs_offdp1 = a_bs[iA_offdp1];
      a_bsw_offd = a_bsw[iA_offd];
      a_bsw_offdm1 = a_bsw[iA_offdm1];
      a_bsw_offdp1 = a_bsw[iA_offdp1];
      a_bse_offd = a_bse[iA_offd];
      a_bse_offdm1 = a_bse[iA_offdm1];
      a_bse_offdp1 = a_bse[iA_offdp1];
      a_be_offd = a_be[iA_offd];
      a_be_offdm1 = a_be[iA_offdm1];
      a_bw_offd = a_bw[iA_offd];
      a_bw_offdm1 = a_bw[iA_offdm1];
      a_bw_offdp1 = a_bw[iA_offdp1];
      a_bn_offd = a_bn[iA_offd];
      a_bn_offdm1 = a_bn[iA_offdm1];
      a_bnw_offd = a_bnw[iA_offd];
      a_bnw_offdm1 = a_bnw[iA_offdm1];
      a_bne_offd = a_bne[iA_offd];
      a_bne_offdm1 = a_bne[iA_offdm1];

#define DEVICE_VAR is_device_ptr(rap_bsw,rb,pa,rap_bs,rap_bse,rap_bw,rap_bc,a_cc,rap_be,rap_bnw,rap_bn,rap_bne,rap_csw,pb,ra,rap_cs,rap_cse,rap_cw,rap_cc)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAm1 = iA - zOffsetA_diag;
         HYPRE_Int iAp1 = iA + zOffsetA_diag;

         HYPRE_Int iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
         rap_bsw[iAc] = rb[iR - rbOffset] * a_csw_offdm1 * pa[iP1]
                        +           rb[iR - rbOffset] * a_bsw_offdm1
                        +                    a_bsw_offd   * pa[iP1];

         iP1 = iP - zOffsetP - yOffsetP;
         rap_bs[iAc] = rb[iR - rbOffset] * a_cs_offdm1 * pa[iP1]
                       +          rb[iR - rbOffset] * a_bs_offdm1
                       +                   a_bs_offd   * pa[iP1];

         iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
         rap_bse[iAc] = rb[iR - rbOffset] * a_cse_offdm1 * pa[iP1]
                        +           rb[iR - rbOffset] * a_bse_offdm1
                        +                    a_bse_offd   * pa[iP1];

         iP1 = iP - zOffsetP - xOffsetP;
         rap_bw[iAc] = rb[iR - rbOffset] * a_cw_offdm1 * pa[iP1]
                       +          rb[iR - rbOffset] * a_bw_offdm1
                       +                   a_bw_offd   * pa[iP1];

         iP1 = iP - zOffsetP;
         rap_bc[iAc] =          a_bc_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pa[iP1]
                                +          rb[iR - rbOffset] * a_bc_offdm1;

         iP1 = iP - zOffsetP + xOffsetP;
         rap_be[iAc] = rb[iR - rbOffset] * a_ce_offdm1 * pa[iP1]
                       +          rb[iR - rbOffset] * a_be_offdm1
                       +                   a_be_offd   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
         rap_bnw[iAc] = rb[iR - rbOffset] * a_cnw_offdm1 * pa[iP1]
                        +           rb[iR - rbOffset] * a_bnw_offdm1
                        +                    a_bnw_offd   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP;
         rap_bn[iAc] = rb[iR - rbOffset] * a_cn_offdm1 * pa[iP1]
                       +          rb[iR - rbOffset] * a_bn_offdm1
                       +                   a_bn_offd   * pa[iP1];

         iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
         rap_bne[iAc] = rb[iR - rbOffset] * a_cne_offdm1 * pa[iP1]
                        +           rb[iR - rbOffset] * a_bne_offdm1
                        +                    a_bne_offd   * pa[iP1];

         iP1 = iP - yOffsetP - xOffsetP;
         rap_csw[iAc] =          a_csw_offd
                                 +          rb[iR - rbOffset] * a_csw_offdm1 * pb[iP1 - pbOffset]
                                 +          ra[iR] * a_csw_offdp1 * pa[iP1]
                                 +                   a_bsw_offd   * pb[iP1 - pbOffset]
                                 +                   a_asw_offd   * pa[iP1]
                                 +          rb[iR - rbOffset] * a_asw_offdm1
                                 +          ra[iR] * a_bsw_offdp1;

         iP1 = iP - yOffsetP;
         rap_cs[iAc] =          a_cs_offd
                                +          rb[iR - rbOffset] * a_cs_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cs_offdp1 * pa[iP1]
                                +                   a_bs_offd   * pb[iP1 - pbOffset]
                                +                   a_as_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_as_offdm1
                                +          ra[iR] * a_bs_offdp1;

         iP1 = iP - yOffsetP + xOffsetP;
         rap_cse[iAc] =          a_cse_offd
                                 +          rb[iR - rbOffset] * a_cse_offdm1 * pb[iP1 - pbOffset]
                                 +          ra[iR] * a_cse_offdp1 * pa[iP1]
                                 +                   a_bse_offd   * pb[iP1 - pbOffset]
                                 +                   a_ase_offd   * pa[iP1]
                                 +          rb[iR - rbOffset] * a_ase_offdm1
                                 +          ra[iR] * a_bse_offdp1;

         iP1 = iP - xOffsetP;
         rap_cw[iAc] =          a_cw_offd
                                +          rb[iR - rbOffset] * a_cw_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cw_offdp1 * pa[iP1]
                                +                   a_bw_offd   * pb[iP1 - pbOffset]
                                +                   a_aw_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_aw_offdm1
                                +          ra[iR] * a_bw_offdp1;

         rap_cc[iAc] =          a_cc[iA]
                                +          rb[iR - rbOffset] * a_cc[iAm1] * pb[iP - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pa[iP]
                                +          rb[iR - rbOffset] * a_ac_offdm1
                                +          ra[iR] * a_bc_offdp1
                                +                   a_bc_offd   * pb[iP - pbOffset]
                                +                   a_ac_offd   * pa[iP];
      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPSym, for one box, one value of fine_stencil_size
   (27) and one value of constant_coefficient (1).  */
HYPRE_Int
hypre_PFMG3BuildRAPSym_onebox_FSS27_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           fstart;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw, *a_as;
   HYPRE_Real           *a_bc, *a_bw, *a_be, *a_bs, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   HYPRE_Real           *a_asw, *a_ase;
   HYPRE_Real           *a_bsw, *a_bse, *a_bnw, *a_bne;
   HYPRE_Real           *rap_cc, *rap_cw, *rap_cs;
   HYPRE_Real           *rap_bc, *rap_bw, *rap_be, *rap_bs, *rap_bn;
   HYPRE_Real           *rap_csw, *rap_cse;
   HYPRE_Real           *rap_bsw, *rap_bse, *rap_bnw, *rap_bne;
   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_bw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   a_bs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point fine grid operator:
    *
    * a_asw is pointer for southwest coefficient in plane above
    * a_ase is pointer for southeast coefficient in plane above
    * a_anw is pointer for northwest coefficient in plane above
    * a_ane is pointer for northeast coefficient in plane above
    * a_bsw is pointer for southwest coefficient in plane below
    * a_bse is pointer for southeast coefficient in plane below
    * a_bnw is pointer for northwest coefficient in plane below
    * a_bne is pointer for northeast coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_asw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_ase = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, -1);
   MapIndex(index_temp, cdir, index);
   a_bsw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, -1);
   MapIndex(index_temp, cdir, index);
   a_bse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the lower triangular part (plus diagonal).
    *
    * rap_cc is pointer for center coefficient (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_cw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bc = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_bw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   rap_be = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bs = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_csw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point coarse grid operator:
    *
    * A 27-point coarse grid operator is produced when the fine grid
    * stencil is 19 or 27 point.
    *
    * We build only the lower triangular part.
    *
    * rap_csw is pointer for southwest coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bsw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bse = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, -1);
   MapIndex(index_temp, cdir, index);
   rap_bne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetA = 0;
   zOffsetP = 0;

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = 0;

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*--------------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for symmetric 27-point fine grid operator; produces a
    * symmetric 27-point coarse grid operator. We calculate only the
    * lower triangular stencil entries: (below-southwest, below-south,
    * below-southeast, below-west, below-center, below-east,
    * below-northwest, below-north, below-northeast, center-southwest,
    * center-south, center-southeast, center-west, and center-center).
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - zOffsetA;
   iAp1 = iA + zOffsetA;

   iP1 = iP - zOffsetP - yOffsetP - xOffsetP;
   rap_bsw[iAc] = rb[iR] * a_csw[iAm1] * pa[iP1]
                  +           rb[iR] * a_bsw[iAm1]
                  +                    a_bsw[iA]   * pa[iP1];

   iP1 = iP - zOffsetP - yOffsetP;
   rap_bs[iAc] = rb[iR] * a_cs[iAm1] * pa[iP1]
                 +          rb[iR] * a_bs[iAm1]
                 +                   a_bs[iA]   * pa[iP1];

   iP1 = iP - zOffsetP - yOffsetP + xOffsetP;
   rap_bse[iAc] = rb[iR] * a_cse[iAm1] * pa[iP1]
                  +           rb[iR] * a_bse[iAm1]
                  +                    a_bse[iA]   * pa[iP1];

   iP1 = iP - zOffsetP - xOffsetP;
   rap_bw[iAc] = rb[iR] * a_cw[iAm1] * pa[iP1]
                 +          rb[iR] * a_bw[iAm1]
                 +                   a_bw[iA]   * pa[iP1];

   iP1 = iP - zOffsetP;
   rap_bc[iAc] =          a_bc[iA]   * pa[iP1]
                          +          rb[iR] * a_cc[iAm1] * pa[iP1]
                          +          rb[iR] * a_bc[iAm1];

   iP1 = iP - zOffsetP + xOffsetP;
   rap_be[iAc] = rb[iR] * a_ce[iAm1] * pa[iP1]
                 +          rb[iR] * a_be[iAm1]
                 +                   a_be[iA]   * pa[iP1];

   iP1 = iP - zOffsetP + yOffsetP - xOffsetP;
   rap_bnw[iAc] = rb[iR] * a_cnw[iAm1] * pa[iP1]
                  +           rb[iR] * a_bnw[iAm1]
                  +                    a_bnw[iA]   * pa[iP1];

   iP1 = iP - zOffsetP + yOffsetP;
   rap_bn[iAc] = rb[iR] * a_cn[iAm1] * pa[iP1]
                 +          rb[iR] * a_bn[iAm1]
                 +                   a_bn[iA]   * pa[iP1];

   iP1 = iP - zOffsetP + yOffsetP + xOffsetP;
   rap_bne[iAc] = rb[iR] * a_cne[iAm1] * pa[iP1]
                  +           rb[iR] * a_bne[iAm1]
                  +                    a_bne[iA]   * pa[iP1];

   iP1 = iP - yOffsetP - xOffsetP;
   rap_csw[iAc] =          a_csw[iA]
                           +          rb[iR] * a_csw[iAm1] * pb[iP1]
                           +          ra[iR] * a_csw[iAp1] * pa[iP1]
                           +                   a_bsw[iA]   * pb[iP1]
                           +                   a_asw[iA]   * pa[iP1]
                           +          rb[iR] * a_asw[iAm1]
                           +          ra[iR] * a_bsw[iAp1];

   iP1 = iP - yOffsetP;
   rap_cs[iAc] =          a_cs[iA]
                          +          rb[iR] * a_cs[iAm1] * pb[iP1]
                          +          ra[iR] * a_cs[iAp1] * pa[iP1]
                          +                   a_bs[iA]   * pb[iP1]
                          +                   a_as[iA]   * pa[iP1]
                          +          rb[iR] * a_as[iAm1]
                          +          ra[iR] * a_bs[iAp1];

   iP1 = iP - yOffsetP + xOffsetP;
   rap_cse[iAc] =          a_cse[iA]
                           +          rb[iR] * a_cse[iAm1] * pb[iP1]
                           +          ra[iR] * a_cse[iAp1] * pa[iP1]
                           +                   a_bse[iA]   * pb[iP1]
                           +                   a_ase[iA]   * pa[iP1]
                           +          rb[iR] * a_ase[iAm1]
                           +          ra[iR] * a_bse[iAp1];

   iP1 = iP - xOffsetP;
   rap_cw[iAc] =          a_cw[iA]
                          +          rb[iR] * a_cw[iAm1] * pb[iP1]
                          +          ra[iR] * a_cw[iAp1] * pa[iP1]
                          +                   a_bw[iA]   * pb[iP1]
                          +                   a_aw[iA]   * pa[iP1]
                          +          rb[iR] * a_aw[iAm1]
                          +          ra[iR] * a_bw[iAp1];

   rap_cc[iAc] =          a_cc[iA]
                          +          rb[iR] * a_cc[iAm1] * pb[iP]
                          +          ra[iR] * a_cc[iAp1] * pa[iP]
                          +          rb[iR] * a_ac[iAm1]
                          +          ra[iR] * a_bc[iAp1]
                          +                   a_bc[iA]   * pb[iP]
                          +                   a_ac[iA]   * pa[iP];

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/*--------------------------------------------------------------------------
 *--------------------------------------------------------------------------*/

HYPRE_Int
hypre_PFMG3BuildRAPNoSym( hypre_StructMatrix *A,
                          hypre_StructMatrix *P,
                          hypre_StructMatrix *R,
                          HYPRE_Int           cdir,
                          hypre_Index         cindex,
                          hypre_Index         cstride,
                          hypre_StructMatrix *RAP     )
{
   hypre_StructStencil  *fine_stencil;
   HYPRE_Int             fine_stencil_size;

   hypre_StructGrid     *fgrid;
   HYPRE_Int            *fgrid_ids;
   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   HYPRE_Int            *cgrid_ids;
   HYPRE_Int             fi, ci;
   HYPRE_Int             constant_coefficient;
   HYPRE_Int             constant_coefficient_A;

   fine_stencil = hypre_StructMatrixStencil(A);
   fine_stencil_size = hypre_StructStencilSize(fine_stencil);

   fgrid = hypre_StructMatrixGrid(A);
   fgrid_ids = hypre_StructGridIDs(fgrid);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);
   cgrid_ids = hypre_StructGridIDs(cgrid);

   constant_coefficient = hypre_StructMatrixConstantCoefficient(RAP);
   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);
   hypre_assert( constant_coefficient == 0 || constant_coefficient == 1 );
   hypre_assert( hypre_StructMatrixConstantCoefficient(R) == constant_coefficient );
   hypre_assert( hypre_StructMatrixConstantCoefficient(P) == constant_coefficient );
   if (constant_coefficient == 1 )
   {
      hypre_assert( constant_coefficient_A == 1 );
   }
   else
   {
      hypre_assert( constant_coefficient_A == 0 || constant_coefficient_A == 2 );
   }

   fi = 0;
   hypre_ForBoxI(ci, cgrid_boxes)
   {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
         fi++;
      }

      switch (fine_stencil_size)
      {

         /*--------------------------------------------------------------
          * Loop for 7-point fine grid operator; produces upper triangular
          * part of 19-point coarse grid operator. stencil entries:
          * (above-north, above-east, above-center, above-west,
          * above-south, center-north, and center-east).
          *--------------------------------------------------------------*/

         case 7:

            if ( constant_coefficient == 1 )
            {
               hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            else
            {
               hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

         /*--------------------------------------------------------------
          * Loop for 19-point fine grid operator; produces upper triangular
          * part of 27-point coarse grid operator. stencil entries:
          * (above-northeast, above-north, above-northwest, above-east,
          * above-center, above-west, above-southeast, above-south,
          * above-southwest, center-northeast, center-north,
          * center-northwest, and center-east).
          *--------------------------------------------------------------*/

         case 19:

            if ( constant_coefficient == 1 )
            {
               hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            else
            {
               hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

         /*--------------------------------------------------------------
          * Loop for 27-point fine grid operator; produces upper triangular
          * part of 27-point coarse grid operator. stencil entries:
          * (above-northeast, above-north, above-northwest, above-east,
          * above-center, above-west, above-southeast, above-south,
          * above-southwest, center-northeast, center-north,
          * center-northwest, and center-east).
          *--------------------------------------------------------------*/

         default:

            if ( constant_coefficient == 1 )
            {
               hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC1(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }
            else
            {
               hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC0(
                  ci, fi, A, P, R, cdir, cindex, cstride, RAP );
            }

            break;

      } /* end switch statement */

   } /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPNoSym, for one box, one value of fine_stencil_size
   (07) and one value of constant_coefficient (0).  */
HYPRE_Int
hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient_A;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;
   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;
   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac;
   HYPRE_Real            a_cn_offd, a_cn_offdm1, a_cn_offdp1;
   HYPRE_Real            a_ce_offd, a_ce_offdm1, a_ce_offdp1;
   HYPRE_Real            a_cs_offdp1, a_cw_offdp1;
   HYPRE_Real            a_ac_offd, a_ac_offdp1;
   HYPRE_Real           *rap_ce, *rap_cn;
   HYPRE_Real           *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   HYPRE_Real           *rap_cnw, *rap_cne;
   HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             zOffsetA_diag;
   HYPRE_Int             zOffsetA_offd;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

   /* fi = 0;
      hypre_ForBoxI(ci, cgrid_boxes)
      {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
      fi++;
      }
   */
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   HYPRE_Int pbOffset = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET
   HYPRE_Int rbOffset = hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the upper triangular part (excluding diagonal).
    *
    * rap_ce is pointer for east coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ac = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_aw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ae = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_as = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_an = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      zOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      zOffsetA_diag = hypre_BoxOffsetDistance(A_dbox, index);
      zOffsetA_offd = 0;
   }

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for 7-point fine grid operator; produces upper triangular
    * part of 19-point coarse grid operator. stencil entries:
    * (above-north, above-east, above-center, above-west,
    * above-south, center-north, and center-east).
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
#define DEVICE_VAR is_device_ptr(rap_an,ra,a_cn,pb,rap_ae,a_ce,rap_ac,a_ac,a_cc,rap_aw,a_cw,rap_as,a_cs,rap_cn,rb,pa,rap_ce,rap_cnw,rap_cne)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAm1 = iA - zOffsetA;
         HYPRE_Int iAp1 = iA + zOffsetA;

         HYPRE_Int iP1 = iP + zOffsetP + yOffsetP;
         rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + xOffsetP;
         rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP;
         rap_ac[iAc] =          a_ac[iA]   * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ac[iAp1];

         iP1 = iP + zOffsetP - xOffsetP;
         rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP;
         rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] =          a_cn[iA]
                                +          rb[iR - rbOffset] * a_cn[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cn[iAp1] * pa[iP1];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce[iA]
                                +          rb[iR - rbOffset] * a_ce[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce[iAp1] * pa[iP1];

         rap_cnw[iAc] = 0.0;

         rap_cne[iAc] = 0.0;
      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      iA_offd = 0;
      iA_offdm1 = iA_offd - zOffsetA_offd;
      iA_offdp1 = iA_offd + zOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cn_offdp1 = a_cn[iA_offdp1];
      a_ce_offd = a_ce[iA_offd];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_ce_offdp1 = a_ce[iA_offdp1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_ac_offd   = a_ac[iA_offd];
      a_ac_offdp1 = a_ac[iA_offdp1];

#define DEVICE_VAR is_device_ptr(rap_an,ra,pb,rap_ae,rap_ac,a_cc,rap_aw,rap_as,rap_cn,rb,pa,rap_ce,rap_cnw,rap_cne)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         //HYPRE_Int iAm1 = iA - zOffsetA_diag;
         HYPRE_Int iAp1 = iA + zOffsetA_diag;

         HYPRE_Int iP1 = iP + zOffsetP + yOffsetP;
         rap_an[iAc] = ra[iR] * a_cn_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + xOffsetP;
         rap_ae[iAc] = ra[iR] * a_ce_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP;
         rap_ac[iAc] =          a_ac_offd   * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ac_offdp1;

         iP1 = iP + zOffsetP - xOffsetP;
         rap_aw[iAc] = ra[iR] * a_cw_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP;
         rap_as[iAc] = ra[iR] * a_cs_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] =          a_cn_offd
                                +          rb[iR - rbOffset] * a_cn_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cn_offdp1 * pa[iP1];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce_offd
                                +          rb[iR - rbOffset] * a_ce_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce_offdp1 * pa[iP1];

         rap_cnw[iAc] = 0.0;

         rap_cne[iAc] = 0.0;
      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPNoSym, for one box, one value of fine_stencil_size
   (07) and one value of constant_coefficient (1).  */
HYPRE_Int
hypre_PFMG3BuildRAPNoSym_onebox_FSS07_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           fstart;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac;
   HYPRE_Real           *rap_ce, *rap_cn;
   HYPRE_Real           *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   HYPRE_Real           *rap_cnw, *rap_cne;
   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
   HYPRE_Int             zOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the upper triangular part (excluding diagonal).
    *
    * rap_ce is pointer for east coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ac = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_aw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ae = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_as = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_an = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetA = 0;
   zOffsetP = 0;

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = 0;

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for 7-point fine grid operator; produces upper triangular
    * part of 19-point coarse grid operator. stencil entries:
    * (above-north, above-east, above-center, above-west,
    * above-south, center-north, and center-east).
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - zOffsetA;
   iAp1 = iA + zOffsetA;

   iP1 = iP + zOffsetP + yOffsetP;
   rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1];

   iP1 = iP + zOffsetP + xOffsetP;
   rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1];

   iP1 = iP + zOffsetP;
   rap_ac[iAc] =          a_ac[iA]   * pb[iP1]
                          +          ra[iR] * a_cc[iAp1] * pb[iP1]
                          +          ra[iR] * a_ac[iAp1];

   iP1 = iP + zOffsetP - xOffsetP;
   rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1];

   iP1 = iP + zOffsetP - yOffsetP;
   rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1];

   iP1 = iP + yOffsetP;
   rap_cn[iAc] =          a_cn[iA]
                          +          rb[iR] * a_cn[iAm1] * pb[iP1]
                          +          ra[iR] * a_cn[iAp1] * pa[iP1];

   iP1 = iP + xOffsetP;
   rap_ce[iAc] =          a_ce[iA]
                          +          rb[iR] * a_ce[iAm1] * pb[iP1]
                          +          ra[iR] * a_ce[iAp1] * pa[iP1];

   rap_cnw[iAc] = 0.0;

   rap_cne[iAc] = 0.0;

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPNoSym, for one box, one value of fine_stencil_size
   (19) and one value of constant_coefficient (0).  */
HYPRE_Int
hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient_A;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;
   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw, *a_ae, *a_as, *a_an;
   HYPRE_Real           *a_be, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   HYPRE_Real            a_cn_offd, a_cn_offdm1, a_cn_offdp1;
   HYPRE_Real            a_ce_offd, a_ce_offdm1, a_ce_offdp1;
   HYPRE_Real            a_cs_offdp1, a_cw_offdp1, a_cse_offdp1, a_csw_offdp1;
   HYPRE_Real            a_cne_offd, a_cne_offdm1, a_cne_offdp1;
   HYPRE_Real            a_cnw_offd, a_cnw_offdm1, a_cnw_offdp1;
   HYPRE_Real            a_ac_offd, a_ac_offdp1;
   HYPRE_Real            a_an_offd, a_an_offdm1, a_an_offdp1;
   HYPRE_Real            a_as_offd, a_as_offdp1;
   HYPRE_Real            a_aw_offd, a_aw_offdp1;
   HYPRE_Real            a_ae_offd, a_ae_offdm1, a_ae_offdp1;
   HYPRE_Real            a_be_offd, a_be_offdp1;
   HYPRE_Real            a_bn_offd, a_bn_offdp1;
   HYPRE_Real           *rap_ce, *rap_cn;
   HYPRE_Real           *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   HYPRE_Real           *rap_cnw, *rap_cne;
   HYPRE_Real           *rap_asw, *rap_ase, *rap_anw, *rap_ane;
   HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;
   HYPRE_Int             zOffsetA;
   HYPRE_Int             zOffsetA_diag;
   HYPRE_Int             zOffsetA_offd;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

   /* fi = 0;
      hypre_ForBoxI(ci, cgrid_boxes)
      {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
      fi++;
      }
   */
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   HYPRE_Int pbOffset = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET
   HYPRE_Int rbOffset = hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the upper triangular part (excluding diagonal).
    *
    * rap_ce is pointer for east coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ac = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_aw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ae = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_as = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_an = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point coarse grid operator:
    *
    * A 27-point coarse grid operator is produced when the fine grid
    * stencil is 19 or 27 point.
    *
    * We build only the upper triangular part.
    *
    * rap_cnw is pointer for northwest coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_asw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_ase = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_anw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_ane = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      zOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      zOffsetA_diag = hypre_BoxOffsetDistance(A_dbox, index);
      zOffsetA_offd = 0;
   }

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for 19-point fine grid operator; produces upper triangular
    * part of 27-point coarse grid operator. stencil entries:
    * (above-northeast, above-north, above-northwest, above-east,
    * above-center, above-west, above-southeast, above-south,
    * above-southwest, center-northeast, center-north,
    * center-northwest, and center-east).
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
#define DEVICE_VAR is_device_ptr(rap_ane,ra,a_cne,pb,rap_an,a_cn,a_an,rap_anw,a_cnw,rap_ae,a_ce,a_ae,rap_ac,a_ac,a_cc,rap_aw,a_cw,a_aw,rap_ase,a_cse,rap_as,a_cs,a_as,rap_asw,a_csw,rap_cne,rb,pa,rap_cn,a_bn,rap_cnw,rap_ce,a_be)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAm1 = iA - zOffsetA;
         HYPRE_Int iAp1 = iA + zOffsetA;

         HYPRE_Int iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
         rap_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + yOffsetP;
         rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_an[iAp1]
                       +                   a_an[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
         rap_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + xOffsetP;
         rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_ae[iAp1]
                       +                   a_ae[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP;
         rap_ac[iAc] =          a_ac[iA]   * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ac[iAp1];

         iP1 = iP + zOffsetP - xOffsetP;
         rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_aw[iAp1]
                       +                   a_aw[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
         rap_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP;
         rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_as[iAp1]
                       +                   a_as[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
         rap_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP + xOffsetP;
         rap_cne[iAc] =         a_cne[iA]
                                +          rb[iR - rbOffset] * a_cne[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cne[iAp1] * pa[iP1];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] =          a_cn[iA]
                                +          rb[iR - rbOffset] * a_cn[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cn[iAp1] * pa[iP1]
                                +                   a_bn[iA]   * pb[iP1 - pbOffset]
                                +                   a_an[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_an[iAm1]
                                +          ra[iR] * a_bn[iAp1];

         iP1 = iP + yOffsetP - xOffsetP;
         rap_cnw[iAc] =         a_cnw[iA]
                                +          rb[iR - rbOffset] * a_cnw[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cnw[iAp1] * pa[iP1];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce[iA]
                                +          rb[iR - rbOffset] * a_ce[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce[iAp1] * pa[iP1]
                                +                   a_be[iA]   * pb[iP1 - pbOffset]
                                +                   a_ae[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_ae[iAm1]
                                +          ra[iR] * a_be[iAp1];

      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      iA_offd = 0;
      iA_offdm1 = iA_offd - zOffsetA_offd;
      iA_offdp1 = iA_offd + zOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cn_offdp1 = a_cn[iA_offdp1];
      a_cne_offd = a_cne[iA_offd];
      a_cne_offdm1 = a_cne[iA_offdm1];
      a_cne_offdp1 = a_cne[iA_offdp1];
      a_cnw_offd = a_cnw[iA_offd];
      a_cnw_offdm1 = a_cnw[iA_offdm1];
      a_cnw_offdp1 = a_cnw[iA_offdp1];
      a_ce_offd = a_ce[iA_offd];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_ce_offdp1 = a_ce[iA_offdp1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cse_offdp1 = a_cse[iA_offdp1];
      a_csw_offdp1 = a_csw[iA_offdp1];
      a_ac_offd = a_ac[iA_offd];
      a_ac_offdp1 = a_ac[iA_offdp1];
      a_an_offd = a_an[iA_offd];
      a_an_offdm1 = a_an[iA_offdm1];
      a_an_offdp1 = a_an[iA_offdp1];
      a_as_offd = a_as[iA_offd];
      a_as_offdp1 = a_as[iA_offdp1];
      a_aw_offd = a_aw[iA_offd];
      a_aw_offdp1 = a_aw[iA_offdp1];
      a_ae_offd = a_ae[iA_offd];
      a_ae_offdm1 = a_ae[iA_offdm1];
      a_ae_offdp1 = a_ae[iA_offdp1];
      a_be_offd = a_be[iA_offd];
      a_be_offdp1 = a_be[iA_offdp1];
      a_bn_offd = a_bn[iA_offd];
      a_bn_offdp1 = a_bn[iA_offdp1];

#define DEVICE_VAR is_device_ptr(rap_ane,ra,pb,rap_an,rap_anw,rap_ae,rap_ac,a_cc,rap_aw,rap_ase,rap_as,rap_asw,rap_cne,rb,pa,rap_cn,rap_cnw,rap_ce)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAp1 = iA + zOffsetA_diag;

         HYPRE_Int iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
         rap_ane[iAc] = ra[iR] * a_cne_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + yOffsetP;
         rap_an[iAc] = ra[iR] * a_cn_offdp1 * pb[iP1 - pbOffset]
                       +          ra[iR] * a_an_offdp1
                       +                   a_an_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
         rap_anw[iAc] = ra[iR] * a_cnw_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + xOffsetP;
         rap_ae[iAc] = ra[iR] * a_ce_offdp1 * pb[iP1 - pbOffset]
                       +          ra[iR] * a_ae_offdp1
                       +                   a_ae_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP;
         rap_ac[iAc] =          a_ac_offd   * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ac_offdp1;

         iP1 = iP + zOffsetP - xOffsetP;
         rap_aw[iAc] = ra[iR] * a_cw_offdp1 * pb[iP1 - pbOffset]
                       +          ra[iR] * a_aw_offdp1
                       +                   a_aw_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
         rap_ase[iAc] = ra[iR] * a_cse_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP;
         rap_as[iAc] = ra[iR] * a_cs_offdp1 * pb[iP1 - pbOffset]
                       +          ra[iR] * a_as_offdp1
                       +                   a_as_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
         rap_asw[iAc] = ra[iR] * a_csw_offdp1 * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP + xOffsetP;
         rap_cne[iAc] =         a_cne_offd
                                +          rb[iR - rbOffset] * a_cne_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cne_offdp1 * pa[iP1];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] =          a_cn_offd
                                +          rb[iR - rbOffset] * a_cn_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cn_offdp1 * pa[iP1]
                                +                   a_bn_offd   * pb[iP1 - pbOffset]
                                +                   a_an_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_an_offdm1
                                +          ra[iR] * a_bn_offdp1;

         iP1 = iP + yOffsetP - xOffsetP;
         rap_cnw[iAc] =         a_cnw_offd
                                +          rb[iR - rbOffset] * a_cnw_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cnw_offdp1 * pa[iP1];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce_offd
                                +          rb[iR - rbOffset] * a_ce_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce_offdp1 * pa[iP1]
                                +                   a_be_offd   * pb[iP1 - pbOffset]
                                +                   a_ae_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_ae_offdm1
                                +          ra[iR] * a_be_offdp1;

      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPNoSym, for one box, one value of fine_stencil_size
   (19) and one value of constant_coefficient (1).  */
HYPRE_Int
hypre_PFMG3BuildRAPNoSym_onebox_FSS19_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           fstart;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;
   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw, *a_ae, *a_as, *a_an;
   HYPRE_Real           *a_be, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   HYPRE_Real           *rap_ce, *rap_cn;
   HYPRE_Real           *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   HYPRE_Real           *rap_cnw, *rap_cne;
   HYPRE_Real           *rap_asw, *rap_ase, *rap_anw, *rap_ane;
   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;
   HYPRE_Int             zOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the upper triangular part (excluding diagonal).
    *
    * rap_ce is pointer for east coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ac = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_aw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ae = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_as = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_an = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point coarse grid operator:
    *
    * A 27-point coarse grid operator is produced when the fine grid
    * stencil is 19 or 27 point.
    *
    * We build only the upper triangular part.
    *
    * rap_cnw is pointer for northwest coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_asw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_ase = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_anw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_ane = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetA = 0;
   zOffsetP = 0;

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = 0;

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for 19-point fine grid operator; produces upper triangular
    * part of 27-point coarse grid operator. stencil entries:
    * (above-northeast, above-north, above-northwest, above-east,
    * above-center, above-west, above-southeast, above-south,
    * above-southwest, center-northeast, center-north,
    * center-northwest, and center-east).
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - zOffsetA;
   iAp1 = iA + zOffsetA;

   iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
   rap_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1];

   iP1 = iP + zOffsetP + yOffsetP;
   rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                 +          ra[iR] * a_an[iAp1]
                 +                   a_an[iA]   * pb[iP1];

   iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
   rap_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1];

   iP1 = iP + zOffsetP + xOffsetP;
   rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                 +          ra[iR] * a_ae[iAp1]
                 +                   a_ae[iA]   * pb[iP1];

   iP1 = iP + zOffsetP;
   rap_ac[iAc] =          a_ac[iA]   * pb[iP1]
                          +          ra[iR] * a_cc[iAp1] * pb[iP1]
                          +          ra[iR] * a_ac[iAp1];

   iP1 = iP + zOffsetP - xOffsetP;
   rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                 +          ra[iR] * a_aw[iAp1]
                 +                   a_aw[iA]   * pb[iP1];

   iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
   rap_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1];

   iP1 = iP + zOffsetP - yOffsetP;
   rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                 +          ra[iR] * a_as[iAp1]
                 +                   a_as[iA]   * pb[iP1];

   iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
   rap_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1];

   iP1 = iP + yOffsetP + xOffsetP;
   rap_cne[iAc] =         a_cne[iA]
                          +          rb[iR] * a_cne[iAm1] * pb[iP1]
                          +          ra[iR] * a_cne[iAp1] * pa[iP1];

   iP1 = iP + yOffsetP;
   rap_cn[iAc] =          a_cn[iA]
                          +          rb[iR] * a_cn[iAm1] * pb[iP1]
                          +          ra[iR] * a_cn[iAp1] * pa[iP1]
                          +                   a_bn[iA]   * pb[iP1]
                          +                   a_an[iA]   * pa[iP1]
                          +          rb[iR] * a_an[iAm1]
                          +          ra[iR] * a_bn[iAp1];

   iP1 = iP + yOffsetP - xOffsetP;
   rap_cnw[iAc] =         a_cnw[iA]
                          +          rb[iR] * a_cnw[iAm1] * pb[iP1]
                          +          ra[iR] * a_cnw[iAp1] * pa[iP1];

   iP1 = iP + xOffsetP;
   rap_ce[iAc] =          a_ce[iA]
                          +          rb[iR] * a_ce[iAm1] * pb[iP1]
                          +          ra[iR] * a_ce[iAp1] * pa[iP1]
                          +                   a_be[iA]   * pb[iP1]
                          +                   a_ae[iA]   * pa[iP1]
                          +          rb[iR] * a_ae[iAm1]
                          +          ra[iR] * a_be[iAp1];

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPNoSym, for one box, one value of fine_stencil_size
   (27) and one value of constant_coefficient (0).  */
HYPRE_Int
hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC0(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           stridec;
   hypre_Index           fstart;
   hypre_IndexRef        stridef;
   hypre_Index           loop_size;

   HYPRE_Int             constant_coefficient_A;

   hypre_Box            *A_dbox;
   hypre_Box            *P_dbox;
   hypre_Box            *R_dbox;
   hypre_Box            *RAP_dbox;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;

   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw, *a_ae, *a_as, *a_an;
   HYPRE_Real           *a_be, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   HYPRE_Real           *a_asw, *a_ase, *a_anw, *a_ane;
   HYPRE_Real           *a_bnw, *a_bne;
   HYPRE_Real            a_cn_offd, a_cn_offdm1, a_cn_offdp1;
   HYPRE_Real            a_ce_offd, a_ce_offdm1, a_ce_offdp1;
   HYPRE_Real            a_cs_offdp1, a_cw_offdp1, a_cse_offdp1, a_csw_offdp1;
   HYPRE_Real            a_cne_offd, a_cne_offdm1, a_cne_offdp1;
   HYPRE_Real            a_cnw_offd, a_cnw_offdm1, a_cnw_offdp1;
   HYPRE_Real            a_ac_offd, a_ac_offdp1;
   HYPRE_Real            a_an_offd, a_an_offdm1, a_an_offdp1;
   HYPRE_Real            a_ane_offd, a_ane_offdm1, a_ane_offdp1;
   HYPRE_Real            a_anw_offd, a_anw_offdm1, a_anw_offdp1;
   HYPRE_Real            a_as_offd, a_as_offdp1;
   HYPRE_Real            a_ase_offd, a_ase_offdp1, a_asw_offd, a_asw_offdp1;
   HYPRE_Real            a_aw_offd, a_aw_offdp1;
   HYPRE_Real            a_ae_offd, a_ae_offdm1, a_ae_offdp1;
   HYPRE_Real            a_be_offd, a_be_offdp1;
   HYPRE_Real            a_bn_offd, a_bn_offdp1;
   HYPRE_Real            a_bne_offd, a_bne_offdp1, a_bnw_offd, a_bnw_offdp1;

   HYPRE_Real           *rap_ce, *rap_cn;
   HYPRE_Real           *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   HYPRE_Real           *rap_cnw, *rap_cne;
   HYPRE_Real           *rap_asw, *rap_ase, *rap_anw, *rap_ane;

   HYPRE_Int             iA_offd, iA_offdm1, iA_offdp1;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             zOffsetA_diag;
   HYPRE_Int             zOffsetA_offd;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   stridef = cstride;
   hypre_SetIndex3(stridec, 1, 1, 1);

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   constant_coefficient_A = hypre_StructMatrixConstantCoefficient(A);

   /* fi = 0;
      hypre_ForBoxI(ci, cgrid_boxes)
      {
      while (fgrid_ids[fi] != cgrid_ids[ci])
      {
      fi++;
      }
   */
   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   A_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(A), fi);
   P_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(P), fi);
   R_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(R), fi);
   RAP_dbox = hypre_BoxArrayBox(hypre_StructMatrixDataSpace(RAP), ci);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);
   //RL PTROFFSET
   HYPRE_Int pbOffset = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);
   //RL PTROFFSET
   HYPRE_Int rbOffset = hypre_BoxOffsetDistance(R_dbox, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point fine grid operator:
    *
    * a_asw is pointer for southwest coefficient in plane above
    * a_ase is pointer for southeast coefficient in plane above
    * a_anw is pointer for northwest coefficient in plane above
    * a_ane is pointer for northeast coefficient in plane above
    * a_bsw is pointer for southwest coefficient in plane below
    * a_bse is pointer for southeast coefficient in plane below
    * a_bnw is pointer for northwest coefficient in plane below
    * a_bne is pointer for northeast coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_asw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_ase = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 1);
   MapIndex(index_temp, cdir, index);
   a_anw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 1);
   MapIndex(index_temp, cdir, index);
   a_ane = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the upper triangular part (excluding diagonal).
    *
    * rap_ce is pointer for east coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ac = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_aw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ae = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_as = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_an = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point coarse grid operator:
    *
    * A 27-point coarse grid operator is produced when the fine grid
    * stencil is 19 or 27 point.
    *
    * We build only the upper triangular part.
    *
    * rap_cnw is pointer for northwest coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_asw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_ase = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_anw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_ane = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetP = hypre_BoxOffsetDistance(P_dbox, index);
   if ( constant_coefficient_A == 0 )
   {
      zOffsetA = hypre_BoxOffsetDistance(A_dbox, index);
   }
   else
   {
      zOffsetA_diag = hypre_BoxOffsetDistance(A_dbox, index);
      zOffsetA_offd = 0;
   }

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = hypre_BoxOffsetDistance(P_dbox, index);

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for 27-point fine grid operator; produces upper triangular
    * part of 27-point coarse grid operator. stencil entries:
    * (above-northeast, above-north, above-northwest, above-east,
    * above-center, above-west, above-southeast, above-south,
    * above-southwest, center-northeast, center-north,
    * center-northwest, and center-east).
    *--------------------------------------------------------------*/

   hypre_BoxGetSize(cgrid_box, loop_size);

   if ( constant_coefficient_A == 0 )
   {
#define DEVICE_VAR is_device_ptr(rap_ane,ra,a_cne,pb,a_ane,rap_an,a_cn,a_an,rap_anw,a_cnw,a_anw,rap_ae,a_ce,a_ae,rap_ac,a_ac,a_cc,rap_aw,a_cw,a_aw,rap_ase,a_cse,a_ase,rap_as,a_cs,a_as,rap_asw,a_csw,a_asw,rap_cne,rb,pa,a_bne,rap_cn,a_bn,rap_cnw,a_bnw,rap_ce,a_be)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAm1 = iA - zOffsetA;
         HYPRE_Int iAp1 = iA + zOffsetA;

         HYPRE_Int iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
         rap_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1 - pbOffset]
                        +           ra[iR] * a_ane[iAp1]
                        +                    a_ane[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + yOffsetP;
         rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_an[iAp1]
                       +                   a_an[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
         rap_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1 - pbOffset]
                        +           ra[iR] * a_anw[iAp1]
                        +                    a_anw[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + xOffsetP;
         rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_ae[iAp1]
                       +                   a_ae[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP;
         rap_ac[iAc] =          a_ac[iA]   * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ac[iAp1];

         iP1 = iP + zOffsetP - xOffsetP;
         rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_aw[iAp1]
                       +                   a_aw[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
         rap_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1 - pbOffset]
                        +           ra[iR] * a_ase[iAp1]
                        +                    a_ase[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP;
         rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1 - pbOffset]
                       +          ra[iR] * a_as[iAp1]
                       +                   a_as[iA]   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
         rap_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1 - pbOffset]
                        +           ra[iR] * a_asw[iAp1]
                        +                    a_asw[iA]   * pb[iP1 - pbOffset];


         iP1 = iP + yOffsetP + xOffsetP;
         rap_cne[iAc] =         a_cne[iA]
                                +          rb[iR - rbOffset] * a_cne[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cne[iAp1] * pa[iP1]
                                +                   a_bne[iA]   * pb[iP1 - pbOffset]
                                +                   a_ane[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_ane[iAm1]
                                +          ra[iR] * a_bne[iAp1];

         iP1 = iP + yOffsetP;
         rap_cn[iAc] =          a_cn[iA]
                                +          rb[iR - rbOffset] * a_cn[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cn[iAp1] * pa[iP1]
                                +                   a_bn[iA]   * pb[iP1 - pbOffset]
                                +                   a_an[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_an[iAm1]
                                +          ra[iR] * a_bn[iAp1];

         iP1 = iP + yOffsetP - xOffsetP;
         rap_cnw[iAc] =         a_cnw[iA]
                                +          rb[iR - rbOffset] * a_cnw[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cnw[iAp1] * pa[iP1]
                                +                   a_bnw[iA]   * pb[iP1 - pbOffset]
                                +                   a_anw[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_anw[iAm1]
                                +          ra[iR] * a_bnw[iAp1];

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce[iA]
                                +          rb[iR - rbOffset] * a_ce[iAm1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce[iAp1] * pa[iP1]
                                +                   a_be[iA]   * pb[iP1 - pbOffset]
                                +                   a_ae[iA]   * pa[iP1]
                                +          rb[iR - rbOffset] * a_ae[iAm1]
                                +          ra[iR] * a_be[iAp1];

      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }
   else
   {
      iA_offd = 0;
      iA_offdm1 = iA_offd - zOffsetA_offd;
      iA_offdp1 = iA_offd + zOffsetA_offd;
      a_cn_offd = a_cn[iA_offd];
      a_cn_offdm1 = a_cn[iA_offdm1];
      a_cn_offdp1 = a_cn[iA_offdp1];
      a_cne_offd = a_cne[iA_offd];
      a_cne_offdm1 = a_cne[iA_offdm1];
      a_cne_offdp1 = a_cne[iA_offdp1];
      a_cnw_offd = a_cnw[iA_offd];
      a_cnw_offdm1 = a_cnw[iA_offdm1];
      a_cnw_offdp1 = a_cnw[iA_offdp1];
      a_ce_offd = a_ce[iA_offd];
      a_ce_offdm1 = a_ce[iA_offdm1];
      a_ce_offdp1 = a_ce[iA_offdp1];
      a_cs_offdp1 = a_cs[iA_offdp1];
      a_cse_offdp1 = a_cse[iA_offdp1];
      a_csw_offdp1 = a_csw[iA_offdp1];
      a_cw_offdp1 = a_cw[iA_offdp1];
      a_ac_offd = a_ac[iA_offd];
      a_ac_offdp1 = a_ac[iA_offdp1];
      a_an_offd = a_an[iA_offd];
      a_an_offdm1 = a_an[iA_offdm1];
      a_an_offdp1 = a_an[iA_offdp1];
      a_ane_offd = a_ane[iA_offd];
      a_ane_offdm1 = a_ane[iA_offdm1];
      a_ane_offdp1 = a_ane[iA_offdp1];
      a_anw_offd = a_anw[iA_offd];
      a_anw_offdm1 = a_anw[iA_offdm1];
      a_anw_offdp1 = a_anw[iA_offdp1];
      a_ae_offd = a_ae[iA_offd];
      a_ae_offdm1 = a_ae[iA_offdm1];
      a_ae_offdp1 = a_ae[iA_offdp1];
      a_aw_offd = a_aw[iA_offd];
      a_aw_offdp1 = a_aw[iA_offdp1];
      a_as_offd = a_as[iA_offd];
      a_as_offdp1 = a_as[iA_offdp1];
      a_ase_offd = a_ase[iA_offd];
      a_ase_offdp1 = a_ase[iA_offdp1];
      a_asw_offd = a_asw[iA_offd];
      a_asw_offdp1 = a_asw[iA_offdp1];
      a_bn_offd = a_bn[iA_offd];
      a_bn_offdp1 = a_bn[iA_offdp1];
      a_bne_offd = a_bne[iA_offd];
      a_bne_offdp1 = a_bne[iA_offdp1];
      a_bnw_offd = a_bnw[iA_offd];
      a_bnw_offdp1 = a_bnw[iA_offdp1];
      a_be_offd = a_be[iA_offd];
      a_be_offdp1 = a_be[iA_offdp1];

#define DEVICE_VAR is_device_ptr(rap_ane,ra,pb,rap_an,rap_anw,rap_ae,rap_ac,a_cc,rap_aw,rap_ase,rap_as,rap_asw,rap_cne,rb,pa,rap_cn,rap_cnw,rap_ce)
      hypre_BoxLoop4Begin(hypre_StructMatrixNDim(A), loop_size,
                          P_dbox, cstart, stridec, iP,
                          R_dbox, cstart, stridec, iR,
                          A_dbox, fstart, stridef, iA,
                          RAP_dbox, cstart, stridec, iAc);
      {
         HYPRE_Int iAp1 = iA + zOffsetA_diag;

         HYPRE_Int iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
         rap_ane[iAc] = ra[iR] * a_cne_offdp1 * pb[iP1 - pbOffset]
                        +           ra[iR] * a_ane_offdp1
                        +                    a_ane_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + yOffsetP;
         rap_an[iAc] = ra[iR] * a_cn_offdp1 * pb[iP1 - pbOffset]
                       +          ra[iR] * a_an_offdp1
                       +                   a_an_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
         rap_anw[iAc] = ra[iR] * a_cnw_offdp1 * pb[iP1 - pbOffset]
                        +           ra[iR] * a_anw_offdp1
                        +                    a_anw_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP + xOffsetP;
         rap_ae[iAc] = ra[iR] * a_ce_offdp1 * pb[iP1 - pbOffset]
                       +          ra[iR] * a_ae_offdp1
                       +                   a_ae_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP;
         rap_ac[iAc] =          a_ac_offd   * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cc[iAp1] * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ac_offdp1;

         iP1 = iP + zOffsetP - xOffsetP;
         rap_aw[iAc] = ra[iR] * a_cw_offdp1 * pb[iP1 - pbOffset]
                       +          ra[iR] * a_aw_offdp1
                       +                   a_aw_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
         rap_ase[iAc] = ra[iR] * a_cse_offdp1 * pb[iP1 - pbOffset]
                        +           ra[iR] * a_ase_offdp1
                        +                    a_ase_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP;
         rap_as[iAc] = ra[iR] * a_cs_offdp1 * pb[iP1 - pbOffset]
                       +          ra[iR] * a_as_offdp1
                       +                   a_as_offd   * pb[iP1 - pbOffset];

         iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
         rap_asw[iAc] = ra[iR] * a_csw_offdp1 * pb[iP1 - pbOffset]
                        +           ra[iR] * a_asw_offdp1
                        +                    a_asw_offd   * pb[iP1 - pbOffset];

         iP1 = iP + yOffsetP + xOffsetP;
         rap_cne[iAc] =         a_cne_offd
                                +          rb[iR - rbOffset] * a_cne_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cne_offdp1 * pa[iP1]
                                +                   a_bne_offd   * pb[iP1 - pbOffset]
                                +                   a_ane_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_ane_offdm1
                                +          ra[iR] * a_bne_offdp1;

         iP1 = iP + yOffsetP;
         rap_cn[iAc] =          a_cn_offd
                                +          rb[iR - rbOffset] * a_cn_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cn_offdp1 * pa[iP1]
                                +                   a_bn_offd   * pb[iP1 - pbOffset]
                                +                   a_an_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_an_offdm1
                                +          ra[iR] * a_bn_offdp1;

         iP1 = iP + yOffsetP - xOffsetP;
         rap_cnw[iAc] =         a_cnw_offd
                                +          rb[iR - rbOffset] * a_cnw_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_cnw_offdp1 * pa[iP1]
                                +                   a_bnw_offd   * pb[iP1 - pbOffset]
                                +                   a_anw_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_anw_offdm1
                                +          ra[iR] * a_bnw_offdp1;

         iP1 = iP + xOffsetP;
         rap_ce[iAc] =          a_ce_offd
                                +          rb[iR - rbOffset] * a_ce_offdm1 * pb[iP1 - pbOffset]
                                +          ra[iR] * a_ce_offdp1 * pa[iP1]
                                +                   a_be_offd   * pb[iP1 - pbOffset]
                                +                   a_ae_offd   * pa[iP1]
                                +          rb[iR - rbOffset] * a_ae_offdm1
                                +          ra[iR] * a_be_offdp1;

      }
      hypre_BoxLoop4End(iP, iR, iA, iAc);
#undef DEVICE_VAR
   }

   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}

/* core part of hypre_PFMG3BuildRAPNoSym, for one box, one value of fine_stencil_size
   (27) and one value of constant_coefficient (1).  */
HYPRE_Int
hypre_PFMG3BuildRAPNoSym_onebox_FSS27_CC1(
   HYPRE_Int             ci,
   HYPRE_Int             fi,
   hypre_StructMatrix *A,
   hypre_StructMatrix *P,
   hypre_StructMatrix *R,
   HYPRE_Int           cdir,
   hypre_Index         cindex,
   hypre_Index         cstride,
   hypre_StructMatrix *RAP     )
{

   hypre_Index           index;
   hypre_Index           index_temp;

   hypre_StructGrid     *cgrid;
   hypre_BoxArray       *cgrid_boxes;
   hypre_Box            *cgrid_box;
   hypre_IndexRef        cstart;
   hypre_Index           fstart;

   HYPRE_Real           *pa, *pb;
   HYPRE_Real           *ra, *rb;
   HYPRE_Real           *a_cc, *a_cw, *a_ce, *a_cs, *a_cn;
   HYPRE_Real           *a_ac, *a_aw, *a_ae, *a_as, *a_an;
   HYPRE_Real           *a_be, *a_bn;
   HYPRE_Real           *a_csw, *a_cse, *a_cnw, *a_cne;
   HYPRE_Real           *a_asw, *a_ase, *a_anw, *a_ane;
   HYPRE_Real           *a_bnw, *a_bne;
   HYPRE_Real           *rap_ce, *rap_cn;
   HYPRE_Real           *rap_ac, *rap_aw, *rap_ae, *rap_as, *rap_an;
   HYPRE_Real           *rap_cnw, *rap_cne;
   HYPRE_Real           *rap_asw, *rap_ase, *rap_anw, *rap_ane;
   HYPRE_Int             iA, iAm1, iAp1;
   HYPRE_Int             iAc;
   HYPRE_Int             iP, iP1;
   HYPRE_Int             iR;

   HYPRE_Int             zOffsetA;
   HYPRE_Int             xOffsetP;
   HYPRE_Int             yOffsetP;
   HYPRE_Int             zOffsetP;

   cgrid = hypre_StructMatrixGrid(RAP);
   cgrid_boxes = hypre_StructGridBoxes(cgrid);

   cgrid_box = hypre_BoxArrayBox(cgrid_boxes, ci);

   cstart = hypre_BoxIMin(cgrid_box);
   hypre_StructMapCoarseToFine(cstart, cindex, cstride, fstart);

   /*-----------------------------------------------------------------
    * Extract pointers for interpolation operator:
    * pa is pointer for weight for f-point above c-point
    * pb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   pa = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   pb = hypre_StructMatrixExtractPointerByIndex(P, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for restriction operator:
    * ra is pointer for weight for f-point above c-point
    * rb is pointer for weight for f-point below c-point
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, -1);
   MapIndex(index_temp, cdir, index);
   ra = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   rb = hypre_StructMatrixExtractPointerByIndex(R, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 7-point fine grid operator:
    *
    * a_cc is pointer for center coefficient
    * a_cw is pointer for west coefficient in same plane
    * a_ce is pointer for east coefficient in same plane
    * a_cs is pointer for south coefficient in same plane
    * a_cn is pointer for north coefficient in same plane
    * a_ac is pointer for center coefficient in plane above
    * a_bc is pointer for center coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cc = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_cw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   a_ce = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cs = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ac = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 19-point fine grid operator:
    *
    * a_aw is pointer for west coefficient in plane above
    * a_ae is pointer for east coefficient in plane above
    * a_as is pointer for south coefficient in plane above
    * a_an is pointer for north coefficient in plane above
    * a_bw is pointer for west coefficient in plane below
    * a_be is pointer for east coefficient in plane below
    * a_bs is pointer for south coefficient in plane below
    * a_bn is pointer for north coefficient in plane below
    * a_csw is pointer for southwest coefficient in same plane
    * a_cse is pointer for southeast coefficient in same plane
    * a_cnw is pointer for northwest coefficient in same plane
    * a_cne is pointer for northeast coefficient in same plane
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_aw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   a_ae = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_as = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   a_an = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 0, -1);
   MapIndex(index_temp, cdir, index);
   a_be = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 0, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bn = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_csw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 0);
   MapIndex(index_temp, cdir, index);
   a_cse = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   a_cne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point fine grid operator:
    *
    * a_asw is pointer for southwest coefficient in plane above
    * a_ase is pointer for southeast coefficient in plane above
    * a_anw is pointer for northwest coefficient in plane above
    * a_ane is pointer for northeast coefficient in plane above
    * a_bsw is pointer for southwest coefficient in plane below
    * a_bse is pointer for southeast coefficient in plane below
    * a_bnw is pointer for northwest coefficient in plane below
    * a_bne is pointer for northeast coefficient in plane below
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_asw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, -1, 1);
   MapIndex(index_temp, cdir, index);
   a_ase = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, 1);
   MapIndex(index_temp, cdir, index);
   a_anw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, 1);
   MapIndex(index_temp, cdir, index);
   a_ane = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, -1, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bnw = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   hypre_SetIndex3(index_temp, 1, 1, -1);
   MapIndex(index_temp, cdir, index);
   a_bne = hypre_StructMatrixExtractPointerByIndex(A, fi, index);

   /*-----------------------------------------------------------------
    * Extract pointers for 19-point coarse grid operator:
    *
    * We build only the upper triangular part (excluding diagonal).
    *
    * rap_ce is pointer for east coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);
   rap_ce = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cn = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ac = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_aw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 0, 1);
   MapIndex(index_temp, cdir, index);
   rap_ae = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_as = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 0, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_an = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cnw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 0);
   MapIndex(index_temp, cdir, index);
   rap_cne = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Extract additional pointers for 27-point coarse grid operator:
    *
    * A 27-point coarse grid operator is produced when the fine grid
    * stencil is 19 or 27 point.
    *
    * We build only the upper triangular part.
    *
    * rap_cnw is pointer for northwest coefficient in same plane (etc.)
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, -1, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_asw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, -1, 1);
   MapIndex(index_temp, cdir, index);
   rap_ase = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, -1, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_anw = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   hypre_SetIndex3(index_temp, 1, 1, 1);
   MapIndex(index_temp, cdir, index);
   rap_ane = hypre_StructMatrixExtractPointerByIndex(RAP, ci, index);

   /*-----------------------------------------------------------------
    * Define offsets for fine grid stencil and interpolation
    *
    * In the BoxLoop below I assume iA and iP refer to data associated
    * with the point which we are building the stencil for. The below
    * Offsets are used in refering to data associated with other points.
    *-----------------------------------------------------------------*/

   hypre_SetIndex3(index_temp, 0, 0, 1);
   MapIndex(index_temp, cdir, index);

   zOffsetA = 0;
   zOffsetP = 0;

   hypre_SetIndex3(index_temp, 0, 1, 0);
   MapIndex(index_temp, cdir, index);

   yOffsetP = 0;

   hypre_SetIndex3(index_temp, 1, 0, 0);
   MapIndex(index_temp, cdir, index);

   xOffsetP = 0;

   /*-----------------------------------------------------------------
    * Switch statement to direct control to apropriate BoxLoop depending
    * on stencil size. Default is full 27-point.
    *-----------------------------------------------------------------*/

   /*--------------------------------------------------------------
    * Loop for 27-point fine grid operator; produces upper triangular
    * part of 27-point coarse grid operator. stencil entries:
    * (above-northeast, above-north, above-northwest, above-east,
    * above-center, above-west, above-southeast, above-south,
    * above-southwest, center-northeast, center-north,
    * center-northwest, and center-east).
    *--------------------------------------------------------------*/

   iP = 0;
   iR = 0;
   iA = 0;
   iAc = 0;

   iAm1 = iA - zOffsetA;
   iAp1 = iA + zOffsetA;

   iP1 = iP + zOffsetP + yOffsetP + xOffsetP;
   rap_ane[iAc] = ra[iR] * a_cne[iAp1] * pb[iP1]
                  +           ra[iR] * a_ane[iAp1]
                  +                    a_ane[iA]   * pb[iP1];

   iP1 = iP + zOffsetP + yOffsetP;
   rap_an[iAc] = ra[iR] * a_cn[iAp1] * pb[iP1]
                 +          ra[iR] * a_an[iAp1]
                 +                   a_an[iA]   * pb[iP1];

   iP1 = iP + zOffsetP + yOffsetP - xOffsetP;
   rap_anw[iAc] = ra[iR] * a_cnw[iAp1] * pb[iP1]
                  +           ra[iR] * a_anw[iAp1]
                  +                    a_anw[iA]   * pb[iP1];

   iP1 = iP + zOffsetP + xOffsetP;
   rap_ae[iAc] = ra[iR] * a_ce[iAp1] * pb[iP1]
                 +          ra[iR] * a_ae[iAp1]
                 +                   a_ae[iA]   * pb[iP1];

   iP1 = iP + zOffsetP;
   rap_ac[iAc] =          a_ac[iA]   * pb[iP1]
                          +          ra[iR] * a_cc[iAp1] * pb[iP1]
                          +          ra[iR] * a_ac[iAp1];

   iP1 = iP + zOffsetP - xOffsetP;
   rap_aw[iAc] = ra[iR] * a_cw[iAp1] * pb[iP1]
                 +          ra[iR] * a_aw[iAp1]
                 +                   a_aw[iA]   * pb[iP1];

   iP1 = iP + zOffsetP - yOffsetP + xOffsetP;
   rap_ase[iAc] = ra[iR] * a_cse[iAp1] * pb[iP1]
                  +           ra[iR] * a_ase[iAp1]
                  +                    a_ase[iA]   * pb[iP1];

   iP1 = iP + zOffsetP - yOffsetP;
   rap_as[iAc] = ra[iR] * a_cs[iAp1] * pb[iP1]
                 +          ra[iR] * a_as[iAp1]
                 +                   a_as[iA]   * pb[iP1];

   iP1 = iP + zOffsetP - yOffsetP - xOffsetP;
   rap_asw[iAc] = ra[iR] * a_csw[iAp1] * pb[iP1]
                  +           ra[iR] * a_asw[iAp1]
                  +                    a_asw[iA]   * pb[iP1];


   iP1 = iP + yOffsetP + xOffsetP;
   rap_cne[iAc] =         a_cne[iA]
                          +          rb[iR] * a_cne[iAm1] * pb[iP1]
                          +          ra[iR] * a_cne[iAp1] * pa[iP1]
                          +                   a_bne[iA]   * pb[iP1]
                          +                   a_ane[iA]   * pa[iP1]
                          +          rb[iR] * a_ane[iAm1]
                          +          ra[iR] * a_bne[iAp1];

   iP1 = iP + yOffsetP;
   rap_cn[iAc] =          a_cn[iA]
                          +          rb[iR] * a_cn[iAm1] * pb[iP1]
                          +          ra[iR] * a_cn[iAp1] * pa[iP1]
                          +                   a_bn[iA]   * pb[iP1]
                          +                   a_an[iA]   * pa[iP1]
                          +          rb[iR] * a_an[iAm1]
                          +          ra[iR] * a_bn[iAp1];

   iP1 = iP + yOffsetP - xOffsetP;
   rap_cnw[iAc] =         a_cnw[iA]
                          +          rb[iR] * a_cnw[iAm1] * pb[iP1]
                          +          ra[iR] * a_cnw[iAp1] * pa[iP1]
                          +                   a_bnw[iA]   * pb[iP1]
                          +                   a_anw[iA]   * pa[iP1]
                          +          rb[iR] * a_anw[iAm1]
                          +          ra[iR] * a_bnw[iAp1];

   iP1 = iP + xOffsetP;
   rap_ce[iAc] =          a_ce[iA]
                          +          rb[iR] * a_ce[iAm1] * pb[iP1]
                          +          ra[iR] * a_ce[iAp1] * pa[iP1]
                          +                   a_be[iA]   * pb[iP1]
                          +                   a_ae[iA]   * pa[iP1]
                          +          rb[iR] * a_ae[iAm1]
                          +          ra[iR] * a_be[iAp1];


   /*      }*/ /* end ForBoxI */

   return hypre_error_flag;
}
