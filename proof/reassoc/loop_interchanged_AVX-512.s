	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 0
	.file	"loop_interchanged_control.c"
	.def	test_reduction_interchanged;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	test_reduction_interchanged     # -- Begin function test_reduction_interchanged
	.p2align	4
test_reduction_interchanged:            # @test_reduction_interchanged
.seh_proc test_reduction_interchanged
# %bb.0:
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$32, %rsp
	.seh_stackalloc 32
	.seh_endprologue
	testl	%edx, %edx
	jle	.LBB0_15
# %bb.1:
	movq	%r9, %rsi
	movq	%r8, %rdi
	movl	%edx, %ebx
	movl	%ecx, %ebp
	movl	%edx, %r14d
	leaq	(,%r14,4), %r8
	movq	%r9, %rcx
	xorl	%edx, %edx
	callq	memset
	testl	%ebp, %ebp
	jle	.LBB0_15
# %bb.2:
	movl	%ebp, %eax
	movl	%r14d, %ecx
	andl	$2147483584, %ecx               # imm = 0x7FFFFFC0
	movl	%r14d, %edx
	andl	$2147483640, %edx               # imm = 0x7FFFFFF8
	movl	%ebx, %r8d
	shrl	$6, %r8d
	andl	$33554431, %r8d                 # imm = 0x1FFFFFF
	shlq	$8, %r8
	leaq	192(%rdi), %r9
	leaq	(,%r14,4), %r10
	xorl	%r11d, %r11d
	jmp	.LBB0_3
	.p2align	4
.LBB0_14:                               #   in Loop: Header=BB0_3 Depth=1
	incq	%r11
	addq	%r10, %r9
	addq	%r10, %rdi
	cmpq	%rax, %r11
	je	.LBB0_15
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_8 Depth 2
                                        #     Child Loop BB0_11 Depth 2
                                        #     Child Loop BB0_13 Depth 2
	cmpl	$8, %ebx
	jae	.LBB0_5
# %bb.4:                                #   in Loop: Header=BB0_3 Depth=1
	xorl	%r15d, %r15d
	jmp	.LBB0_13
	.p2align	4
.LBB0_5:                                #   in Loop: Header=BB0_3 Depth=1
	cmpl	$64, %ebx
	jae	.LBB0_7
# %bb.6:                                #   in Loop: Header=BB0_3 Depth=1
	xorl	%r12d, %r12d
	jmp	.LBB0_11
	.p2align	4
.LBB0_7:                                #   in Loop: Header=BB0_3 Depth=1
	xorl	%r15d, %r15d
	.p2align	4
.LBB0_8:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovups	-192(%r9,%r15), %zmm0
	vmovups	-128(%r9,%r15), %zmm1
	vmovups	-64(%r9,%r15), %zmm2
	vmovups	(%r9,%r15), %zmm3
	vaddps	(%rsi,%r15), %zmm0, %zmm0
	vaddps	64(%rsi,%r15), %zmm1, %zmm1
	vaddps	128(%rsi,%r15), %zmm2, %zmm2
	vaddps	192(%rsi,%r15), %zmm3, %zmm3
	vmovups	%zmm0, (%rsi,%r15)
	vmovups	%zmm1, 64(%rsi,%r15)
	vmovups	%zmm2, 128(%rsi,%r15)
	vmovups	%zmm3, 192(%rsi,%r15)
	addq	$256, %r15                      # imm = 0x100
	cmpq	%r15, %r8
	jne	.LBB0_8
# %bb.9:                                #   in Loop: Header=BB0_3 Depth=1
	cmpl	%r14d, %ecx
	je	.LBB0_14
# %bb.10:                               #   in Loop: Header=BB0_3 Depth=1
	movq	%rcx, %r12
	movq	%rcx, %r15
	testb	$56, %r14b
	je	.LBB0_13
	.p2align	4
.LBB0_11:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovups	(%rdi,%r12,4), %ymm0
	vaddps	(%rsi,%r12,4), %ymm0, %ymm0
	vmovups	%ymm0, (%rsi,%r12,4)
	addq	$8, %r12
	cmpq	%r12, %rdx
	jne	.LBB0_11
# %bb.12:                               #   in Loop: Header=BB0_3 Depth=1
	movq	%rdx, %r15
	cmpl	%r14d, %edx
	je	.LBB0_14
	.p2align	4
.LBB0_13:                               #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	vmovss	(%rdi,%r15,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	vaddss	(%rsi,%r15,4), %xmm0, %xmm0
	vmovss	%xmm0, (%rsi,%r15,4)
	incq	%r15
	cmpq	%r15, %r14
	jne	.LBB0_13
	jmp	.LBB0_14
.LBB0_15:
	.seh_startepilogue
	addq	$32, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r14
	popq	%r15
	.seh_endepilogue
	vzeroupper
	retq
	.seh_endproc
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
