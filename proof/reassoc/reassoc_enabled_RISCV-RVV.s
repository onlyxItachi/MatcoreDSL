	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"reassoc_enabled.c"
	.text
	.globl	test_reduction_reassoc          # -- Begin function test_reduction_reassoc
	.p2align	1
	.type	test_reduction_reassoc,@function
test_reduction_reassoc:                 # @test_reduction_reassoc
	.cfi_startproc
# %bb.0:
	blez	a0, .LBB0_3
# %bb.1:
	csrr	a3, vlenb
	srli	a4, a3, 1
	bgeu	a0, a4, .LBB0_4
# %bb.2:
	li	t0, 0
	fmv.w.x	fa0, zero
	j	.LBB0_7
.LBB0_3:
	fmv.w.x	fa0, zero
	ret
.LBB0_4:
	slli	a2, a3, 28
	sub	a2, a2, a4
	and	t0, a2, a0
	sub	a5, t0, a4
	divu	a7, a5, a4
	srli	a5, a3, 3
	lui	a6, 524288
	slli	a3, a5, 4
	vsetvli	a4, zero, e32, m2, ta, ma
	vmv.v.x	v8, a6
	vsetvli	a4, zero, e32, m1, ta, ma
	vmv.v.x	v8, a6
	slli	a7, a7, 4
	addi	a7, a7, 16
	mul	a4, a7, a5
	vsetvli	zero, zero, e32, m1, tu, ma
	vmv.s.x	v8, zero
	add	a4, a4, a1
	mv	a5, a1
.LBB0_5:                                # =>This Inner Loop Header: Depth=1
	vl2re32.v	v10, (a5)
	add	a5, a5, a3
	vsetvli	a2, zero, e32, m2, ta, ma
	vfadd.vv	v8, v8, v10
	bne	a5, a4, .LBB0_5
# %bb.6:
	lui	a2, 524288
	vmv.s.x	v10, a2
	vfredosum.vs	v8, v8, v10
	vfmv.f.s	fa0, v8
	beq	t0, a0, .LBB0_9
.LBB0_7:
	slli	t0, t0, 2
	slli	a2, a0, 2
	add	a0, a1, t0
	add	a1, a1, a2
.LBB0_8:                                # =>This Inner Loop Header: Depth=1
	flw	fa5, 0(a0)
	addi	a0, a0, 4
	fadd.s	fa0, fa0, fa5
	bne	a0, a1, .LBB0_8
.LBB0_9:
	ret
.Lfunc_end0:
	.size	test_reduction_reassoc, .Lfunc_end0-test_reduction_reassoc
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 21.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
