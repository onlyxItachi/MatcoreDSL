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
	pushq	%r14
	.seh_pushreg %r14
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
	jle	.LBB0_8
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
	jle	.LBB0_8
# %bb.2:
	movl	%ebp, %eax
	movl	%r14d, %ecx
	andl	$2147483640, %ecx               # imm = 0x7FFFFFF8
	movl	%ebx, %edx
	shrl	$3, %edx
	andl	$268435455, %edx                # imm = 0xFFFFFFF
	shlq	$5, %rdx
	leaq	16(%rdi), %r8
	leaq	(,%r14,4), %r9
	xorl	%r10d, %r10d
	jmp	.LBB0_3
	.p2align	4
.LBB0_7:                                #   in Loop: Header=BB0_3 Depth=1
	incq	%r10
	addq	%r9, %r8
	addq	%r9, %rdi
	cmpq	%rax, %r10
	je	.LBB0_8
.LBB0_3:                                # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_4 Depth 2
                                        #     Child Loop BB0_6 Depth 2
	xorl	%r11d, %r11d
	cmpl	$8, %ebx
	jb	.LBB0_6
	.p2align	4
.LBB0_4:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movups	-16(%r8,%r11), %xmm0
	movups	(%r8,%r11), %xmm1
	movups	(%rsi,%r11), %xmm2
	addps	%xmm0, %xmm2
	movups	16(%rsi,%r11), %xmm0
	addps	%xmm1, %xmm0
	movups	%xmm2, (%rsi,%r11)
	movups	%xmm0, 16(%rsi,%r11)
	addq	$32, %r11
	cmpq	%r11, %rdx
	jne	.LBB0_4
# %bb.5:                                #   in Loop: Header=BB0_3 Depth=1
	movq	%rcx, %r11
	cmpl	%r14d, %ecx
	je	.LBB0_7
	.p2align	4
.LBB0_6:                                #   Parent Loop BB0_3 Depth=1
                                        # =>  This Inner Loop Header: Depth=2
	movss	(%rdi,%r11,4), %xmm0            # xmm0 = mem[0],zero,zero,zero
	addss	(%rsi,%r11,4), %xmm0
	movss	%xmm0, (%rsi,%r11,4)
	incq	%r11
	cmpq	%r11, %r14
	jne	.LBB0_6
	jmp	.LBB0_7
.LBB0_8:
	.seh_startepilogue
	addq	$32, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r14
	.seh_endepilogue
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
