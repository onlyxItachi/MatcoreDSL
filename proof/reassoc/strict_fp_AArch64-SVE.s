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
	mov	w8, w0
	cnth	x10
	cmp	x10, x8
	b.ls	.LBB0_4
// %bb.2:
	movi	d0, #0000000000000000
	mov	x9, xzr
	b	.LBB0_7
.LBB0_3:
	movi	d0, #0000000000000000
                                        // kill: def $s0 killed $s0 killed $z0
	ret
.LBB0_4:
	rdvl	x9, #1
	mov	w11, #2147483640                // =0x7ffffff8
	movi	d0, #0000000000000000
	lsr	x9, x9, #4
	ptrue	p0.s
	mov	x12, x1
	mul	x9, x9, x11
	rdvl	x11, #2
	and	x9, x9, x8
	mov	x13, x9
.LBB0_5:                                // =>This Inner Loop Header: Depth=1
	ldr	z1, [x12]
	subs	x13, x13, x10
	fadda	s0, p0, s0, z1.s
	ldr	z1, [x12, #1, mul vl]
	add	x12, x12, x11
	fadda	s0, p0, s0, z1.s
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
                                        // kill: def $s0 killed $s0 killed $z0
	ret
.Lfunc_end0:
	.size	test_reduction_strict, .Lfunc_end0-test_reduction_strict
	.cfi_endproc
                                        // -- End function
	.ident	"clang version 21.1.8"
	.section	".note.GNU-stack","",@progbits
	.addrsig
