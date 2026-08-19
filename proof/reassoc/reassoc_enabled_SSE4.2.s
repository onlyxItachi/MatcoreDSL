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
	.globl	__xmm@80000000800000008000000080000000 # -- Begin function test_reduction_reassoc
	.section	.rdata,"dr",discard,__xmm@80000000800000008000000080000000
	.p2align	4, 0x0
__xmm@80000000800000008000000080000000:
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.long	0x80000000                      # float -0
	.globl	__xmm@80000000800000008000000000000000
	.section	.rdata,"dr",discard,__xmm@80000000800000008000000000000000
	.p2align	4, 0x0
__xmm@80000000800000008000000000000000:
	.long	0x00000000                      # float 0
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
	cmpl	$8, %ecx
	jae	.LBB0_4
# %bb.3:
	xorps	%xmm0, %xmm0
	xorl	%ecx, %ecx
	jmp	.LBB0_7
.LBB0_1:
	xorps	%xmm0, %xmm0
	retq
.LBB0_4:
	movl	%eax, %ecx
	andl	$2147483640, %ecx               # imm = 0x7FFFFFF8
	movl	%eax, %r8d
	shrl	$3, %r8d
	andl	$268435455, %r8d                # imm = 0xFFFFFFF
	shlq	$5, %r8
	movaps	__xmm@80000000800000008000000080000000(%rip), %xmm0 # xmm0 = [-0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	movaps	__xmm@80000000800000008000000000000000(%rip), %xmm1 # xmm1 = [0.0E+0,-0.0E+0,-0.0E+0,-0.0E+0]
	xorl	%r9d, %r9d
	.p2align	4
.LBB0_5:                                # =>This Inner Loop Header: Depth=1
	movups	(%rdx,%r9), %xmm2
	addps	%xmm2, %xmm1
	movups	16(%rdx,%r9), %xmm2
	addps	%xmm2, %xmm0
	addq	$32, %r9
	cmpq	%r9, %r8
	jne	.LBB0_5
# %bb.6:
	addps	%xmm1, %xmm0
	movshdup	%xmm0, %xmm1                    # xmm1 = xmm0[1,1,3,3]
	addss	%xmm0, %xmm1
	movaps	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2                    # xmm2 = xmm2[1],xmm0[1]
	addss	%xmm1, %xmm2
	shufps	$255, %xmm0, %xmm0              # xmm0 = xmm0[3,3,3,3]
	addss	%xmm2, %xmm0
	cmpl	%eax, %ecx
	je	.LBB0_8
	.p2align	4
.LBB0_7:                                # =>This Inner Loop Header: Depth=1
	addss	(%rdx,%rcx,4), %xmm0
	incq	%rcx
	cmpq	%rcx, %rax
	jne	.LBB0_7
.LBB0_8:
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
