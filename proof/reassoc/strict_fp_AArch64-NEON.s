	.file	"strict_fp.c"
	.text
	.globl	test_reduction_strict           // -- Begin function test_reduction_strict
	.p2align	2
	.type	test_reduction_strict,@function
test_reduction_strict:                  // @test_reduction_strict
	.cfi_startproc
// %bb.0:
	cmp	w0, #1
	b.lt	.LBB0_3
// %bb.1:
	movi	d0, #0000000000000000
	cmp	w0, #8
	mov	w8, w0
	b.hs	.LBB0_4
// %bb.2:
	mov	x9, xzr
	b	.LBB0_7
.LBB0_3:
	movi	d0, #0000000000000000
	ret
.LBB0_4:
	and	x9, x8, #0x7ffffff8
	add	x10, x1, #16
	mov	x11, x9
.LBB0_5:                                // =>This Inner Loop Header: Depth=1
	ldp	q1, q2, [x10, #-16]
	subs	x11, x11, #8
	add	x10, x10, #32
	mov	s3, v1.s[1]
	fadd	s0, s0, s1
	mov	s4, v1.s[2]
	mov	s1, v1.s[3]
	fadd	s0, s0, s3
	mov	s3, v2.s[2]
	fadd	s0, s0, s4
	fadd	s0, s0, s1
	mov	s1, v2.s[1]
	fadd	s0, s0, s2
	fadd	s0, s0, s1
	mov	s1, v2.s[3]
	fadd	s0, s0, s3
	fadd	s0, s0, s1
	b.ne	.LBB0_5
// %bb.6:
	cmp	x9, x8
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
	.size	test_reduction_strict, .Lfunc_end0-test_reduction_strict
	.cfi_endproc
                                        // -- End function
	.ident	"clang version 21.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
