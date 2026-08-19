	.file	"loop_interchanged_control.c"
	.text
	.globl	test_reduction_interchanged     // -- Begin function test_reduction_interchanged
	.p2align	2
	.type	test_reduction_interchanged,@function
test_reduction_interchanged:            // @test_reduction_interchanged
	.cfi_startproc
// %bb.0:
	cmp	w1, #1
	b.lt	.LBB0_11
// %bb.1:
	stp	x29, x30, [sp, #-48]!           // 16-byte Folded Spill
	.cfi_def_cfa_offset 48
	stp	x22, x21, [sp, #16]             // 16-byte Folded Spill
	stp	x20, x19, [sp, #32]             // 16-byte Folded Spill
	mov	x29, sp
	.cfi_def_cfa w29, 48
	.cfi_offset w19, -8
	.cfi_offset w20, -16
	.cfi_offset w21, -24
	.cfi_offset w22, -32
	.cfi_offset w30, -40
	.cfi_offset w29, -48
	mov	w21, w1
	mov	x20, x2
	mov	w22, w0
	ubfiz	x2, x21, #2, #32
	mov	x0, x3
	mov	w1, wzr
	mov	x19, x3
	bl	memset
	cmp	w22, #1
	b.lt	.LBB0_10
// %bb.2:
	rdvl	x9, #1
	mov	w10, #2147483640                // =0x7ffffff8
	mov	w11, w21
	lsr	x9, x9, #4
	ubfiz	x11, x11, #2, #32
	mov	x8, xzr
	mov	w12, w22
	cnth	x13
	rdvl	x14, #2
	mul	x9, x9, x10
	mov	w10, w21
	b	.LBB0_4
.LBB0_3:                                //   in Loop: Header=BB0_4 Depth=1
	add	x8, x8, #1
	add	x20, x20, x11
	cmp	x8, x12
	b.eq	.LBB0_10
.LBB0_4:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_7 Depth 2
                                        //     Child Loop BB0_9 Depth 2
	cmp	x13, x10
	b.ls	.LBB0_6
// %bb.5:                               //   in Loop: Header=BB0_4 Depth=1
	mov	x15, xzr
	b	.LBB0_9
.LBB0_6:                                //   in Loop: Header=BB0_4 Depth=1
	and	x15, x9, x10
	mov	x16, x20
	mov	x17, x19
	mov	x18, x15
.LBB0_7:                                //   Parent Loop BB0_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	z0, [x16]
	ldr	z1, [x17]
	subs	x18, x18, x13
	ldr	z2, [x16, #1, mul vl]
	ldr	z3, [x17, #1, mul vl]
	add	x16, x16, x14
	fadd	z0.s, z0.s, z1.s
	fadd	z1.s, z2.s, z3.s
	str	z0, [x17]
	str	z1, [x17, #1, mul vl]
	add	x17, x17, x14
	b.ne	.LBB0_7
// %bb.8:                               //   in Loop: Header=BB0_4 Depth=1
	cmp	x15, x10
	b.eq	.LBB0_3
.LBB0_9:                                //   Parent Loop BB0_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	s0, [x20, x15, lsl #2]
	ldr	s1, [x19, x15, lsl #2]
	fadd	s0, s0, s1
	str	s0, [x19, x15, lsl #2]
	add	x15, x15, #1
	cmp	x10, x15
	b.ne	.LBB0_9
	b	.LBB0_3
.LBB0_10:
	.cfi_def_cfa wsp, 48
	ldp	x20, x19, [sp, #32]             // 16-byte Folded Reload
	ldp	x22, x21, [sp, #16]             // 16-byte Folded Reload
	ldp	x29, x30, [sp], #48             // 16-byte Folded Reload
	.cfi_def_cfa_offset 0
	.cfi_restore w19
	.cfi_restore w20
	.cfi_restore w21
	.cfi_restore w22
	.cfi_restore w30
	.cfi_restore w29
.LBB0_11:
	ret
.Lfunc_end0:
	.size	test_reduction_interchanged, .Lfunc_end0-test_reduction_interchanged
	.cfi_endproc
                                        // -- End function
	.ident	"clang version 21.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
