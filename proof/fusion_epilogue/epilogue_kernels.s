	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 0
	.file	"epilogue_kernels.c"
	.def	microkernel_separate_relu;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	microkernel_separate_relu       # -- Begin function microkernel_separate_relu
	.p2align	4
microkernel_separate_relu:              # @microkernel_separate_relu
.seh_proc microkernel_separate_relu
# %bb.0:
	subq	$104, %rsp
	.seh_stackalloc 104
	vmovaps	%xmm11, 80(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm11, 80
	vmovaps	%xmm10, 64(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm10, 64
	vmovaps	%xmm9, 48(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm9, 48
	vmovaps	%xmm8, 32(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm8, 32
	vmovaps	%xmm7, 16(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm7, 16
	vmovaps	%xmm6, (%rsp)                   # 16-byte Spill
	.seh_savexmm %xmm6, 0
	.seh_endprologue
	movl	144(%rsp), %eax
	vxorps	%xmm0, %xmm0, %xmm0
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm4, %xmm4, %xmm4
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm8, %xmm8, %xmm8
	vxorps	%xmm6, %xmm6, %xmm6
	testl	%ecx, %ecx
	jle	.LBB0_3
# %bb.1:
	movl	%ecx, %ecx
	shlq	$4, %rcx
	xorl	%r10d, %r10d
	.p2align	4
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	vmovups	(%rdx,%r10,4), %ymm9
	vmovups	32(%rdx,%r10,4), %ymm10
	vbroadcastss	(%r8,%r10), %ymm11
	vfmadd231ps	%ymm11, %ymm9, %ymm6    # ymm6 = (ymm9 * ymm11) + ymm6
	vfmadd231ps	%ymm11, %ymm10, %ymm8   # ymm8 = (ymm10 * ymm11) + ymm8
	vbroadcastss	4(%r8,%r10), %ymm11
	vfmadd231ps	%ymm11, %ymm9, %ymm7    # ymm7 = (ymm9 * ymm11) + ymm7
	vfmadd231ps	%ymm11, %ymm10, %ymm5   # ymm5 = (ymm10 * ymm11) + ymm5
	vbroadcastss	8(%r8,%r10), %ymm11
	vfmadd231ps	%ymm11, %ymm9, %ymm4    # ymm4 = (ymm9 * ymm11) + ymm4
	vfmadd231ps	%ymm11, %ymm10, %ymm3   # ymm3 = (ymm10 * ymm11) + ymm3
	vbroadcastss	12(%r8,%r10), %ymm11
	vfmadd231ps	%ymm9, %ymm11, %ymm2    # ymm2 = (ymm11 * ymm9) + ymm2
	vfmadd231ps	%ymm11, %ymm10, %ymm1   # ymm1 = (ymm10 * ymm11) + ymm1
	addq	$16, %r10
	cmpq	%r10, %rcx
	jne	.LBB0_2
.LBB0_3:
	vmovups	%ymm6, (%r9)
	vmovups	%ymm8, 32(%r9)
	cltq
	vmovups	%ymm7, (%r9,%rax,4)
	vmovups	%ymm5, 32(%r9,%rax,4)
	leal	(%rax,%rax), %ecx
	movslq	%ecx, %rcx
	vmovups	%ymm4, (%r9,%rcx,4)
	vmovups	%ymm3, 32(%r9,%rcx,4)
	leal	(%rax,%rax,2), %edx
	movslq	%edx, %rdx
	vmovups	%ymm2, (%r9,%rdx,4)
	vmovups	%ymm1, 32(%r9,%rdx,4)
	vmovups	(%r9), %ymm3
	vmovups	32(%r9), %ymm4
	vmaxps	%ymm0, %ymm3, %ymm3
	vmaxps	%ymm0, %ymm4, %ymm4
	vmovups	(%r9,%rax,4), %ymm5
	vmovups	32(%r9,%rax,4), %ymm6
	vmaxps	%ymm0, %ymm5, %ymm5
	vmaxps	%ymm0, %ymm6, %ymm6
	vmovups	(%r9,%rcx,4), %ymm7
	vmovups	32(%r9,%rcx,4), %ymm8
	vmaxps	%ymm0, %ymm7, %ymm7
	vmaxps	%ymm0, %ymm8, %ymm8
	vmaxps	%ymm0, %ymm2, %ymm2
	vmaxps	%ymm0, %ymm1, %ymm0
	vmovups	%ymm3, (%r9)
	vmovups	%ymm4, 32(%r9)
	vmovups	%ymm5, (%r9,%rax,4)
	vmovups	%ymm6, 32(%r9,%rax,4)
	vmovups	%ymm7, (%r9,%rcx,4)
	vmovups	%ymm8, 32(%r9,%rcx,4)
	vmovups	%ymm2, (%r9,%rdx,4)
	vmovups	%ymm0, 32(%r9,%rdx,4)
	vmovaps	(%rsp), %xmm6                   # 16-byte Reload
	vmovaps	16(%rsp), %xmm7                 # 16-byte Reload
	vmovaps	32(%rsp), %xmm8                 # 16-byte Reload
	vmovaps	48(%rsp), %xmm9                 # 16-byte Reload
	vmovaps	64(%rsp), %xmm10                # 16-byte Reload
	vmovaps	80(%rsp), %xmm11                # 16-byte Reload
	.seh_startepilogue
	addq	$104, %rsp
	.seh_endepilogue
	vzeroupper
	retq
	.seh_endproc
                                        # -- End function
	.def	microkernel_true_fused_relu;
	.scl	2;
	.type	32;
	.endef
	.globl	microkernel_true_fused_relu     # -- Begin function microkernel_true_fused_relu
	.p2align	4
microkernel_true_fused_relu:            # @microkernel_true_fused_relu
.seh_proc microkernel_true_fused_relu
# %bb.0:
	subq	$104, %rsp
	.seh_stackalloc 104
	vmovaps	%xmm11, 80(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm11, 80
	vmovaps	%xmm10, 64(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm10, 64
	vmovaps	%xmm9, 48(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm9, 48
	vmovaps	%xmm8, 32(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm8, 32
	vmovaps	%xmm7, 16(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm7, 16
	vmovaps	%xmm6, (%rsp)                   # 16-byte Spill
	.seh_savexmm %xmm6, 0
	.seh_endprologue
	movl	144(%rsp), %eax
	vxorps	%xmm0, %xmm0, %xmm0
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm4, %xmm4, %xmm4
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm7, %xmm7, %xmm7
	vxorps	%xmm8, %xmm8, %xmm8
	vxorps	%xmm5, %xmm5, %xmm5
	testl	%ecx, %ecx
	jle	.LBB1_3
# %bb.1:
	movl	%ecx, %ecx
	shlq	$4, %rcx
	xorl	%r10d, %r10d
	.p2align	4
.LBB1_2:                                # =>This Inner Loop Header: Depth=1
	vmovups	(%rdx,%r10,4), %ymm9
	vmovups	32(%rdx,%r10,4), %ymm10
	vbroadcastss	(%r8,%r10), %ymm11
	vfmadd231ps	%ymm11, %ymm9, %ymm5    # ymm5 = (ymm9 * ymm11) + ymm5
	vfmadd231ps	%ymm11, %ymm10, %ymm8   # ymm8 = (ymm10 * ymm11) + ymm8
	vbroadcastss	4(%r8,%r10), %ymm11
	vfmadd231ps	%ymm11, %ymm9, %ymm7    # ymm7 = (ymm9 * ymm11) + ymm7
	vfmadd231ps	%ymm11, %ymm10, %ymm6   # ymm6 = (ymm10 * ymm11) + ymm6
	vbroadcastss	8(%r8,%r10), %ymm11
	vfmadd231ps	%ymm11, %ymm9, %ymm4    # ymm4 = (ymm9 * ymm11) + ymm4
	vfmadd231ps	%ymm11, %ymm10, %ymm3   # ymm3 = (ymm10 * ymm11) + ymm3
	vbroadcastss	12(%r8,%r10), %ymm11
	vfmadd231ps	%ymm9, %ymm11, %ymm2    # ymm2 = (ymm11 * ymm9) + ymm2
	vfmadd231ps	%ymm11, %ymm10, %ymm1   # ymm1 = (ymm10 * ymm11) + ymm1
	addq	$16, %r10
	cmpq	%r10, %rcx
	jne	.LBB1_2
.LBB1_3:
	vmaxps	%ymm0, %ymm5, %ymm5
	vmaxps	%ymm0, %ymm8, %ymm8
	vmaxps	%ymm0, %ymm7, %ymm7
	vmaxps	%ymm0, %ymm6, %ymm6
	vmaxps	%ymm0, %ymm4, %ymm4
	vmaxps	%ymm0, %ymm3, %ymm3
	vmaxps	%ymm0, %ymm2, %ymm2
	vmaxps	%ymm0, %ymm1, %ymm0
	vmovups	%ymm5, (%r9)
	vmovups	%ymm8, 32(%r9)
	cltq
	vmovups	%ymm7, (%r9,%rax,4)
	vmovups	%ymm6, 32(%r9,%rax,4)
	leal	(%rax,%rax), %ecx
	movslq	%ecx, %rcx
	vmovups	%ymm4, (%r9,%rcx,4)
	vmovups	%ymm3, 32(%r9,%rcx,4)
	leal	(%rax,%rax,2), %eax
	cltq
	vmovups	%ymm2, (%r9,%rax,4)
	vmovups	%ymm0, 32(%r9,%rax,4)
	vmovaps	(%rsp), %xmm6                   # 16-byte Reload
	vmovaps	16(%rsp), %xmm7                 # 16-byte Reload
	vmovaps	32(%rsp), %xmm8                 # 16-byte Reload
	vmovaps	48(%rsp), %xmm9                 # 16-byte Reload
	vmovaps	64(%rsp), %xmm10                # 16-byte Reload
	vmovaps	80(%rsp), %xmm11                # 16-byte Reload
	.seh_startepilogue
	addq	$104, %rsp
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
