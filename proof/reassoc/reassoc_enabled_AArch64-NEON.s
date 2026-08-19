	.file	"reassoc_enabled.c"
	.section	.rodata.cst16,"aM",@progbits,16
	.p2align	4, 0x0                          // -- Begin function test_reduction_reassoc
.LCPI0_0:
	.word	0x00000000                      // float 0
	.word	0x80000000                      // float -0
	.word	0x80000000                      // float -0
	.word	0x80000000                      // float -0
	.text
	.globl	test_reduction_reassoc
	.p2align	2
	.type	test_reduction_reassoc,@function
test_reduction_reassoc:                 // @test_reduction_reassoc
	.cfi_startproc
// %bb.0:
	cmp	w0, #1
	b.lt	.LBB0_3
// %bb.1:
	cmp	w0, #8
	mov	w8, w0
	b.hs	.LBB0_4
// %bb.2:
	movi	d0, #0000000000000000
	mov	x9, xzr
	b	.LBB0_7
.LBB0_3:
	movi	d0, #0000000000000000
	ret
.LBB0_4:
	movi	v0.4s, #128, lsl #24
	adrp	x10, .LCPI0_0
	and	x9, x8, #0x7ffffff8
	ldr	q1, [x10, :lo12:.LCPI0_0]
	add	x10, x1, #16
	mov	x11, x9
.LBB0_5:                                // =>This Inner Loop Header: Depth=1
	ldp	q2, q3, [x10, #-16]
	subs	x11, x11, #8
	add	x10, x10, #32
	fadd	v1.4s, v1.4s, v2.4s
	fadd	v0.4s, v0.4s, v3.4s
	b.ne	.LBB0_5
// %bb.6:
	fadd	v0.4s, v0.4s, v1.4s
	cmp	x9, x8
	mov	s1, v0.s[2]
	faddp	s2, v0.2s
	mov	s0, v0.s[3]
	fadd	s1, s2, s1
	fadd	s0, s1, s0
	b.eq	.LBB0_9
.LBB0_7:
	add	x10, x1, x9, lsl #2
	sub	x8, x8, x9
.LBB0_8:                                // =>This Inner Loop Header: Depth=1
	ldr	s1, [x10], #4
	subs	x8, x8, #1
	fadd	s0, s0, s1
	b.ne	.LBB0_8
.LBB0_9:
	ret
.Lfunc_end0:
	.size	test_reduction_reassoc, .Lfunc_end0-test_reduction_reassoc
	.cfi_endproc
                                        // -- End function
	.ident	"clang version 21.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
