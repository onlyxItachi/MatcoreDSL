	.file	"loop_interchanged_control.c"
	.text
	.globl	test_reduction_interchanged     // -- Begin function test_reduction_interchanged
	.p2align	2
	.type	test_reduction_interchanged,@function
test_reduction_interchanged:            // @test_reduction_interchanged
	.cfi_startproc
// %bb.0:
	cmp	w1, #1
	b.lt	.LBB0_12
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
	b.lt	.LBB0_11
// %bb.2:
	mov	w11, w21
	mov	w9, w21
	mov	x8, xzr
	ubfiz	x11, x11, #2, #32
	mov	w10, w22
	and	x12, x9, #0x7ffffff8
	add	x13, x19, #16
	add	x14, x20, #16
	b	.LBB0_4
.LBB0_3:                                //   in Loop: Header=BB0_4 Depth=1
	add	x8, x8, #1
	add	x14, x14, x11
	add	x20, x20, x11
	cmp	x8, x10
	b.eq	.LBB0_11
.LBB0_4:                                // =>This Loop Header: Depth=1
                                        //     Child Loop BB0_7 Depth 2
                                        //     Child Loop BB0_10 Depth 2
	cmp	w21, #8
	b.hs	.LBB0_6
// %bb.5:                               //   in Loop: Header=BB0_4 Depth=1
	mov	x17, xzr
	b	.LBB0_9
.LBB0_6:                                //   in Loop: Header=BB0_4 Depth=1
	mov	x15, x14
	mov	x16, x13
	mov	x17, x12
.LBB0_7:                                //   Parent Loop BB0_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldp	q0, q3, [x16, #-16]
	subs	x17, x17, #8
	ldp	q1, q2, [x15, #-16]
	add	x15, x15, #32
	fadd	v0.4s, v1.4s, v0.4s
	fadd	v1.4s, v2.4s, v3.4s
	stp	q0, q1, [x16, #-16]
	add	x16, x16, #32
	b.ne	.LBB0_7
// %bb.8:                               //   in Loop: Header=BB0_4 Depth=1
	cmp	x12, x9
	mov	x17, x12
	b.eq	.LBB0_3
.LBB0_9:                                //   in Loop: Header=BB0_4 Depth=1
	lsl	x16, x17, #2
	sub	x17, x9, x17
	add	x15, x20, x16
	add	x16, x19, x16
.LBB0_10:                               //   Parent Loop BB0_4 Depth=1
                                        // =>  This Inner Loop Header: Depth=2
	ldr	s0, [x15], #4
	ldr	s1, [x16]
	subs	x17, x17, #1
	fadd	s0, s0, s1
	str	s0, [x16], #4
	b.ne	.LBB0_10
	b	.LBB0_3
.LBB0_11:
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
.LBB0_12:
	ret
.Lfunc_end0:
	.size	test_reduction_interchanged, .Lfunc_end0-test_reduction_interchanged
	.cfi_endproc
                                        // -- End function
	.ident	"clang version 21.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
