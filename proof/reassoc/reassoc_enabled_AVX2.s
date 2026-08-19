	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
@feat.00 = 0
	.file	"reassoc_enabled.c"
	.def	test_reduction_reassoc;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@80000000                 # -- Begin function test_reduction_reassoc
	.section	.rdata,"dr",discard,__real@80000000
	.p2align	2, 0x0
__real@80000000:
	.long	0x80000000                      # float -0
	.globl	__ymm@8000000080000000800000008000000080000000800000008000000000000000
	.section	.rdata,"dr",discard,__ymm@8000000080000000800000008000000080000000800000008000000000000000
	.p2align	5, 0x0
__ymm@8000000080000000800000008000000080000000800000008000000000000000:
	.long	0x00000000                      # float 0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.globl	__xmm@80000000800000008000000000000000
	.section	.rdata,"dr",discard,__xmm@80000000800000008000000000000000
	.p2align	4, 0x0
__xmm@80000000800000008000000000000000:
	.zero	4
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.text
	.globl	test_reduction_reassoc
	.p2align	4
test_reduction_reassoc:                 # @test_reduction_reassoc
# %bb.0:
	testl	%ecx, %ecx
	jle	.LBB0_1
# %bb.2:
	movl	%ecx, %eax
	cmpl	$3, %ecx
	ja	.LBB0_4
# %bb.3:
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB0_13
.LBB0_1:
	vxorps	%xmm0, %xmm0, %xmm0
	retq
.LBB0_4:
	cmpl	$32, %ecx
	jae	.LBB0_6
# %bb.5:
	vxorps	%xmm0, %xmm0, %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB0_10
.LBB0_6:
	movl	%eax, %ecx
	andl	$2147483616, %ecx               # imm = 0x7FFFFFE0
	movl	%eax, %r8d
	shrl	$5, %r8d
	andl	$67108863, %r8d                 # imm = 0x3FFFFFF
	shlq	$7, %r8
	vbroadcastss	__real@80000000(%rip), %ymm0 # ymm0 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	vmovaps	__ymm@8000000080000000800000008000000080000000800000008000000000000000(%rip), %ymm1 # ymm1 = [0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	xorl	%r9d, %r9d
	vmovaps	%ymm0, %ymm2
	vmovaps	%ymm0, %ymm3
	.p2align	4
.LBB0_7:                                # =>This Inner Loop Header: Depth=1
	vaddps	(%rdx,%r9), %ymm1, %ymm1
	vaddps	32(%rdx,%r9), %ymm0, %ymm0
	vaddps	64(%rdx,%r9), %ymm2, %ymm2
	vaddps	96(%rdx,%r9), %ymm3, %ymm3
	subq	$-128, %r9
	cmpq	%r9, %r8
	jne	.LBB0_7
# %bb.8:
	vaddps	%ymm1, %ymm0, %ymm0
	vaddps	%ymm0, %ymm2, %ymm0
	vaddps	%ymm0, %ymm3, %ymm0
	vmovshdup	%xmm0, %xmm1            # xmm1 = xmm0[1,1,3,3]
	vaddss	%xmm1, %xmm0, %xmm1
	vshufpd	$1, %xmm0, %xmm0, %xmm2         # xmm2 = xmm0[1,0]
	vaddss	%xmm2, %xmm1, %xmm1
	vshufps	$255, %xmm0, %xmm0, %xmm2       # xmm2 = xmm0[3,3,3,3]
	vaddss	%xmm2, %xmm1, %xmm1
	vextractf128	$1, %ymm0, %xmm0
	vaddss	%xmm0, %xmm1, %xmm1
	vmovshdup	%xmm0, %xmm2            # xmm2 = xmm0[1,1,3,3]
	vaddss	%xmm2, %xmm1, %xmm1
	vshufpd	$1, %xmm0, %xmm0, %xmm2         # xmm2 = xmm0[1,0]
	vaddss	%xmm2, %xmm1, %xmm1
	vshufps	$255, %xmm0, %xmm0, %xmm0       # xmm0 = xmm0[3,3,3,3]
	vaddss	%xmm0, %xmm1, %xmm0
	cmpl	%eax, %ecx
	je	.LBB0_14
# %bb.9:
	testb	$28, %al
	je	.LBB0_13
.LBB0_10:
	movq	%rcx, %r8
	movl	%eax, %ecx
	andl	$2147483644, %ecx               # imm = 0x7FFFFFFC
	vblendps	$14, __xmm@80000000800000008000000000000000(%rip), %xmm0, %xmm0 # xmm0 = xmm0[0],mem[1,2,3]
	.p2align	4
.LBB0_11:                               # =>This Inner Loop Header: Depth=1
	vaddps	(%rdx,%r8,4), %xmm0, %xmm0
	addq	$4, %r8
	cmpq	%r8, %rcx
	jne	.LBB0_11
# %bb.12:
	vmovshdup	%xmm0, %xmm1            # xmm1 = xmm0[1,1,3,3]
	vaddss	%xmm1, %xmm0, %xmm1
	vshufpd	$1, %xmm0, %xmm0, %xmm2         # xmm2 = xmm0[1,0]
	vaddss	%xmm2, %xmm1, %xmm1
	vshufps	$255, %xmm0, %xmm0, %xmm0       # xmm0 = xmm0[3,3,3,3]
	vaddss	%xmm0, %xmm1, %xmm0
	cmpl	%eax, %ecx
	je	.LBB0_14
	.p2align	4
.LBB0_13:                               # =>This Inner Loop Header: Depth=1
	vaddss	(%rdx,%rcx,4), %xmm0, %xmm0
	incq	%rcx
	cmpq	%rcx, %rax
	jne	.LBB0_13
.LBB0_14:
	vzeroupper
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
