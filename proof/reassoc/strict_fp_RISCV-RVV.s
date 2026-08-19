	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"strict_fp.c"
	.text
	.globl	test_reduction_strict           # -- Begin function test_reduction_strict
	.p2align	1
	.type	test_reduction_strict,@function
test_reduction_strict:                  # @test_reduction_strict
	.cfi_startproc
# %bb.0:
	blez	a0, .LBB0_3
# %bb.1:
	csrr	a3, vlenb
	srli	a4, a3, 1
	bgeu	a0, a4, .LBB0_4
# %bb.2:
	li	a6, 0
	fmv.w.x	fa0, zero
	j	.LBB0_7
.LBB0_3:
	fmv.w.x	fa0, zero
	ret
.LBB0_4:
	slli	a2, a3, 28
	sub	a2, a2, a4
	and	a6, a2, a0
	sub	a5, a6, a4
	divu	a4, a5, a4
	srli	a5, a3, 3
	slli	a3, a5, 4
	slli	a4, a4, 4
	addi	a4, a4, 16
	mul	a4, a4, a5
	vsetvli	a5, zero, e32, m1, ta, ma
	vmv.s.x	v8, zero
	add	a4, a4, a1
	mv	a5, a1
.LBB0_5:                                # =>This Inner Loop Header: Depth=1
	vl2re32.v	v10, (a5)
	add	a5, a5, a3
	vsetvli	a2, zero, e32, m2, ta, ma
	vfredosum.vs	v8, v10, v8
	bne	a5, a4, .LBB0_5
# %bb.6:
	vfmv.f.s	fa0, v8
	beq	a6, a0, .LBB0_9
.LBB0_7:
	slli	a6, a6, 2
	slli	a2, a0, 2
	add	a0, a1, a6
	add	a1, a1, a2
.LBB0_8:                                # =>This Inner Loop Header: Depth=1
	flw	fa5, 0(a0)
	addi	a0, a0, 4
	fadd.s	fa0, fa0, fa5
	bne	a0, a1, .LBB0_8
.LBB0_9:
	ret
.Lfunc_end0:
	.size	test_reduction_strict, .Lfunc_end0-test_reduction_strict
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 21.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
