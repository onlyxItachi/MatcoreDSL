	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 0
	.file	"strict_fp.c"
	.def	test_reduction_strict;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	test_reduction_strict           # -- Begin function test_reduction_strict
	.p2align	4
test_reduction_strict:                  # @test_reduction_strict
# %bb.0:
	testl	%ecx, %ecx
	jle	.LBB0_1
# %bb.2:
	movl	%ecx, %r8d
	movl	%r8d, %eax
	andl	$7, %eax
	cmpl	$8, %ecx
	jae	.LBB0_8
# %bb.3:
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB0_4
.LBB0_1:
	vxorps	%xmm0, %xmm0, %xmm0
	retq
.LBB0_8:
	andl	$2147483640, %r8d               # imm = 0x7FFFFFF8
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%ecx, %ecx
	.p2align	4
.LBB0_9:                                # =>This Inner Loop Header: Depth=1
	vaddss	(%rdx,%rcx,4), %xmm0, %xmm0
	vaddss	4(%rdx,%rcx,4), %xmm0, %xmm0
	vaddss	8(%rdx,%rcx,4), %xmm0, %xmm0
	vaddss	12(%rdx,%rcx,4), %xmm0, %xmm0
	vaddss	16(%rdx,%rcx,4), %xmm0, %xmm0
	vaddss	20(%rdx,%rcx,4), %xmm0, %xmm0
	vaddss	24(%rdx,%rcx,4), %xmm0, %xmm0
	vaddss	28(%rdx,%rcx,4), %xmm0, %xmm0
	addq	$8, %rcx
	cmpq	%rcx, %r8
	jne	.LBB0_9
.LBB0_4:
	testq	%rax, %rax
	je	.LBB0_7
# %bb.5:
	leaq	(%rdx,%rcx,4), %rcx
	xorl	%edx, %edx
	.p2align	4
.LBB0_6:                                # =>This Inner Loop Header: Depth=1
	vaddss	(%rcx,%rdx,4), %xmm0, %xmm0
	incq	%rdx
	cmpq	%rdx, %rax
	jne	.LBB0_6
.LBB0_7:
	retq
                                        # -- End function
	.section	.debug$S,"dr"
	.p2align	2, 0x0
	.long	4                               # Debug section magic
	.long	241
	.long	.Ltmp1-.Ltmp0                   # Subsection size
.Ltmp0:
	.short	.Ltmp3-.Ltmp2                   # Record length
.Ltmp2:
	.short	4353                            # Record kind: S_OBJNAME
	.long	0                               # Signature
	.byte	0                               # Object name
	.p2align	2, 0x0
.Ltmp3:
	.short	.Ltmp5-.Ltmp4                   # Record length
.Ltmp4:
	.short	4412                            # Record kind: S_COMPILE3
	.long	0                               # Flags and language
	.short	208                             # CPUType
	.short	21                              # Frontend version
	.short	1
	.short	8
	.short	0
	.short	21018                           # Backend version
	.short	0
	.short	0
	.short	0
	.asciz	"clang version 21.1.8"          # Null-terminated compiler version string
	.p2align	2, 0x0
.Ltmp5:
.Ltmp1:
	.p2align	2, 0x0
	.addrsig
	.globl	_fltused
