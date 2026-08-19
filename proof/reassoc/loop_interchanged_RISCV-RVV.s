	.attribute	4, 16
	.attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_v1p0_zicsr2p0_zifencei2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0_zve32f1p0_zve32x1p0_zve64d1p0_zve64f1p0_zve64x1p0_zvl128b1p0_zvl32b1p0_zvl64b1p0"
	.file	"loop_interchanged_control.c"
	.text
	.globl	test_reduction_interchanged     # -- Begin function test_reduction_interchanged
	.p2align	1
	.type	test_reduction_interchanged,@function
test_reduction_interchanged:            # @test_reduction_interchanged
	.cfi_startproc
# %bb.0:
	blez	a1, .LBB0_12
# %bb.1:
	addi	sp, sp, -64
	.cfi_def_cfa_offset 64
	sd	ra, 56(sp)                      # 8-byte Folded Spill
	sd	s0, 48(sp)                      # 8-byte Folded Spill
	sd	s1, 40(sp)                      # 8-byte Folded Spill
	sd	s2, 32(sp)                      # 8-byte Folded Spill
	sd	s3, 24(sp)                      # 8-byte Folded Spill
	sd	s4, 16(sp)                      # 8-byte Folded Spill
	sd	s5, 8(sp)                       # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	.cfi_offset s1, -24
	.cfi_offset s2, -32
	.cfi_offset s3, -40
	.cfi_offset s4, -48
	.cfi_offset s5, -56
	mv	s4, a3
	mv	s2, a2
	mv	s5, a1
	mv	s3, a0
	slli	s0, a1, 2
	mv	a0, a3
	li	a1, 0
	mv	a2, s0
	call	memset
	blez	s3, .LBB0_11
# %bb.2:
	li	a0, 0
	csrr	a3, vlenb
	srli	a6, a3, 1
	slli	a1, a3, 28
	neg	a2, a6
	sub	a1, a1, a6
	and	a7, a1, s5
	add	a2, a2, a7
	divu	a1, a2, a6
	srli	a2, a3, 3
	slli	a3, a3, 1
	slli	a1, a1, 4
	addi	a1, a1, 16
	mul	a4, a2, a1
	add	a4, a4, s4
	mv	s1, s2
	j	.LBB0_4
.LBB0_3:                                #   in Loop: Header=BB0_4 Depth=1
	addi	a0, a0, 1
	add	s1, s1, s0
	beq	a0, s3, .LBB0_11
.LBB0_4:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_7 Depth 2
                                        #     Child Loop BB0_10 Depth 2
	bgeu	s5, a6, .LBB0_6
# %bb.5:                                #   in Loop: Header=BB0_4 Depth=1
	li	a1, 0
	j	.LBB0_9
.LBB0_6:                                #   in Loop: Header=BB0_4 Depth=1
	mv	a1, s1
	mv	a2, s4
	vsetvli	a5, zero, e32, m2, ta, ma
.LBB0_7:                                #   Parent Loop BB0_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vl2re32.v	v8, (a1)
	vl2re32.v	v10, (a2)
	vfadd.vv	v8, v8, v10
	vs2r.v	v8, (a2)
	add	a2, a2, a3
	add	a1, a1, a3
	bne	a2, a4, .LBB0_7
# %bb.8:                                #   in Loop: Header=BB0_4 Depth=1
	mv	a1, a7
	beq	a7, s5, .LBB0_3
.LBB0_9:                                #   in Loop: Header=BB0_4 Depth=1
	mul	a5, s0, a0
	slli	a2, a1, 2
	add	a1, s0, a5
	add	a5, s1, a2
	add	a1, a1, s2
	add	a2, a2, s4
.LBB0_10:                               #   Parent Loop BB0_4 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	flw	fa5, 0(a5)
	flw	fa4, 0(a2)
	addi	a5, a5, 4
	fadd.s	fa5, fa5, fa4
	fsw	fa5, 0(a2)
	addi	a2, a2, 4
	bne	a5, a1, .LBB0_10
	j	.LBB0_3
.LBB0_11:
	ld	ra, 56(sp)                      # 8-byte Folded Reload
	ld	s0, 48(sp)                      # 8-byte Folded Reload
	ld	s1, 40(sp)                      # 8-byte Folded Reload
	ld	s2, 32(sp)                      # 8-byte Folded Reload
	ld	s3, 24(sp)                      # 8-byte Folded Reload
	ld	s4, 16(sp)                      # 8-byte Folded Reload
	ld	s5, 8(sp)                       # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	.cfi_restore s1
	.cfi_restore s2
	.cfi_restore s3
	.cfi_restore s4
	.cfi_restore s5
	addi	sp, sp, 64
	.cfi_def_cfa_offset 0
.LBB0_12:
	ret
.Lfunc_end0:
	.size	test_reduction_interchanged, .Lfunc_end0-test_reduction_interchanged
	.cfi_endproc
                                        # -- End function
	.ident	"clang version 21.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
