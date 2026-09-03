	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 0
	.file	"LLVMDialectModule"
	.def	matmul_llvm_16x4_f32;
	.scl	2;
	.type	32;
	.endef
	.text
	.globl	matmul_llvm_16x4_f32            # -- Begin function matmul_llvm_16x4_f32
	.p2align	4
matmul_llvm_16x4_f32:                   # @matmul_llvm_16x4_f32
.seh_proc matmul_llvm_16x4_f32
# %bb.0:
	subq	$136, %rsp
	.seh_stackalloc 136
	vmovaps	%xmm13, 112(%rsp)               # 16-byte Spill
	.seh_savexmm %xmm13, 112
	vmovaps	%xmm12, 96(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm12, 96
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
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%eax, %eax
	xorl	%r9d, %r9d
	vxorps	%xmm1, %xmm1, %xmm1
	vxorps	%xmm2, %xmm2, %xmm2
	vxorps	%xmm3, %xmm3, %xmm3
	vxorps	%xmm4, %xmm4, %xmm4
	vxorps	%xmm5, %xmm5, %xmm5
	vxorps	%xmm6, %xmm6, %xmm6
	vxorps	%xmm7, %xmm7, %xmm7
	cmpq	$63, %r9
	jg	.LBB0_3
	.p2align	4
.LBB0_2:                                # =>This Inner Loop Header: Depth=1
	vmovaps	(%rcx,%rax,2), %ymm8
	vmovaps	32(%rcx,%rax,2), %ymm9
	vbroadcastss	(%rdx,%rax), %ymm10
	vbroadcastss	4(%rdx,%rax), %ymm11
	vbroadcastss	8(%rdx,%rax), %ymm12
	vbroadcastss	12(%rdx,%rax), %ymm13
	vfmadd231ps	%ymm10, %ymm8, %ymm0    # ymm0 = (ymm8 * ymm10) + ymm0
	vfmadd231ps	%ymm10, %ymm9, %ymm1    # ymm1 = (ymm9 * ymm10) + ymm1
	vfmadd231ps	%ymm11, %ymm8, %ymm2    # ymm2 = (ymm8 * ymm11) + ymm2
	vfmadd231ps	%ymm11, %ymm9, %ymm3    # ymm3 = (ymm9 * ymm11) + ymm3
	vfmadd231ps	%ymm12, %ymm8, %ymm4    # ymm4 = (ymm8 * ymm12) + ymm4
	vfmadd231ps	%ymm12, %ymm9, %ymm5    # ymm5 = (ymm9 * ymm12) + ymm5
	vfmadd231ps	%ymm8, %ymm13, %ymm6    # ymm6 = (ymm13 * ymm8) + ymm6
	vfmadd231ps	%ymm13, %ymm9, %ymm7    # ymm7 = (ymm9 * ymm13) + ymm7
	incq	%r9
	addq	$16, %rax
	cmpq	$63, %r9
	jle	.LBB0_2
.LBB0_3:
	vmovaps	%ymm0, (%r8)
	vmovaps	%ymm1, 32(%r8)
	vmovaps	%ymm2, 64(%r8)
	vmovaps	%ymm3, 96(%r8)
	vmovaps	%ymm4, 128(%r8)
	vmovaps	%ymm5, 160(%r8)
	vmovaps	%ymm6, 192(%r8)
	vmovaps	%ymm7, 224(%r8)
	vmovaps	(%rsp), %xmm6                   # 16-byte Reload
	vmovaps	16(%rsp), %xmm7                 # 16-byte Reload
	vmovaps	32(%rsp), %xmm8                 # 16-byte Reload
	vmovaps	48(%rsp), %xmm9                 # 16-byte Reload
	vmovaps	64(%rsp), %xmm10                # 16-byte Reload
	vmovaps	80(%rsp), %xmm11                # 16-byte Reload
	vmovaps	96(%rsp), %xmm12                # 16-byte Reload
	vmovaps	112(%rsp), %xmm13               # 16-byte Reload
	.seh_startepilogue
	addq	$136, %rsp
	.seh_endepilogue
	vzeroupper
	retq
	.seh_endproc
                                        # -- End function
	.globl	_fltused
