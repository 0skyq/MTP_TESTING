�
��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
>
Minimum
x"T
y"T
z"T"
Ttype:
2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
~
CRITIC/dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameCRITIC/dense_7/bias
w
'CRITIC/dense_7/bias/Read/ReadVariableOpReadVariableOpCRITIC/dense_7/bias*
_output_shapes
:*
dtype0
�
CRITIC/dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameCRITIC/dense_7/kernel

)CRITIC/dense_7/kernel/Read/ReadVariableOpReadVariableOpCRITIC/dense_7/kernel*
_output_shapes

:d*
dtype0
~
CRITIC/dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*$
shared_nameCRITIC/dense_6/bias
w
'CRITIC/dense_6/bias/Read/ReadVariableOpReadVariableOpCRITIC/dense_6/bias*
_output_shapes
:d*
dtype0
�
CRITIC/dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�d*&
shared_nameCRITIC/dense_6/kernel
�
)CRITIC/dense_6/kernel/Read/ReadVariableOpReadVariableOpCRITIC/dense_6/kernel*
_output_shapes
:	�d*
dtype0

CRITIC/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameCRITIC/dense_5/bias
x
'CRITIC/dense_5/bias/Read/ReadVariableOpReadVariableOpCRITIC/dense_5/bias*
_output_shapes	
:�*
dtype0
�
CRITIC/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
��*&
shared_nameCRITIC/dense_5/kernel
�
)CRITIC/dense_5/kernel/Read/ReadVariableOpReadVariableOpCRITIC/dense_5/kernel* 
_output_shapes
:
��*
dtype0

CRITIC/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*$
shared_nameCRITIC/dense_4/bias
x
'CRITIC/dense_4/bias/Read/ReadVariableOpReadVariableOpCRITIC/dense_4/bias*
_output_shapes	
:�*
dtype0
�
CRITIC/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d�*&
shared_nameCRITIC/dense_4/kernel
�
)CRITIC/dense_4/kernel/Read/ReadVariableOpReadVariableOpCRITIC/dense_4/kernel*
_output_shapes
:	d�*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:���������d*
dtype0*
shape:���������d
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1CRITIC/dense_4/kernelCRITIC/dense_4/biasCRITIC/dense_5/kernelCRITIC/dense_5/biasCRITIC/dense_6/kernelCRITIC/dense_6/biasCRITIC/dense_7/kernelCRITIC/dense_7/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_13594062

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2


dense3
output_layer
	optimizer
loss

signatures
#_self_saveable_object_factories*
<
0
1
2
3
4
5
6
7*
<
0
1
2
3
4
5
6
7*
* 
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

trace_0
trace_1* 

trace_0
 trace_1* 
* 
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

kernel
bias
#'_self_saveable_object_factories*
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

kernel
bias
#._self_saveable_object_factories*
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

kernel
bias
#5_self_saveable_object_factories*
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

kernel
bias
#<_self_saveable_object_factories*
* 
* 

=serving_default* 
* 
UO
VARIABLE_VALUECRITIC/dense_4/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUECRITIC/dense_4/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUECRITIC/dense_5/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUECRITIC/dense_5/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUECRITIC/dense_6/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUECRITIC/dense_6/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUECRITIC/dense_7/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUECRITIC/dense_7/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
	1

2
3*
* 
* 
* 
* 
* 
* 
* 

0
1*

0
1*
* 
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

Ctrace_0* 

Dtrace_0* 
* 

0
1*

0
1*
* 
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses*

Jtrace_0* 

Ktrace_0* 
* 

0
1*

0
1*
* 
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses*

Qtrace_0* 

Rtrace_0* 
* 

0
1*

0
1*
* 
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses*

Xtrace_0* 

Ytrace_0* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)CRITIC/dense_4/kernel/Read/ReadVariableOp'CRITIC/dense_4/bias/Read/ReadVariableOp)CRITIC/dense_5/kernel/Read/ReadVariableOp'CRITIC/dense_5/bias/Read/ReadVariableOp)CRITIC/dense_6/kernel/Read/ReadVariableOp'CRITIC/dense_6/bias/Read/ReadVariableOp)CRITIC/dense_7/kernel/Read/ReadVariableOp'CRITIC/dense_7/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_13594244
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameCRITIC/dense_4/kernelCRITIC/dense_4/biasCRITIC/dense_5/kernelCRITIC/dense_5/biasCRITIC/dense_6/kernelCRITIC/dense_6/biasCRITIC/dense_7/kernelCRITIC/dense_7/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_13594278��
�	
�
F__inference_dense_23_layer_call_and_return_conditional_losses_13594197

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
+__inference_dense_21_layer_call_fn_13594147

inputs
unknown:
��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_13593891p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_CRITIC_layer_call_and_return_conditional_losses_13594039
input_1$
dense_20_13594018:	d� 
dense_20_13594020:	�%
dense_21_13594023:
�� 
dense_21_13594025:	�$
dense_22_13594028:	�d
dense_22_13594030:d#
dense_23_13594033:d
dense_23_13594035:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��L}
clip_by_value/MinimumMinimuminput_1 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 * ���
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallclip_by_value:z:0dense_20_13594018dense_20_13594020*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_13593874�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_13594023dense_21_13594025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_13593891�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_13594028dense_22_13594030*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_13593908�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_13594033dense_23_13594035*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_13593924x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������d: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_1
�
�
+__inference_dense_23_layer_call_fn_13594187

inputs
unknown:d
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_13593924o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
F__inference_dense_22_layer_call_and_return_conditional_losses_13594178

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
&__inference_signature_wrapper_13594062
input_1
unknown:	d�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�d
	unknown_4:d
	unknown_5:d
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_13593852o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������d: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_1
�
�
+__inference_dense_22_layer_call_fn_13594167

inputs
unknown:	�d
	unknown_0:d
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_13593908o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������d`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_21_layer_call_and_return_conditional_losses_13593891

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�

�
F__inference_dense_20_layer_call_and_return_conditional_losses_13594138

inputs1
matmul_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�

�
F__inference_dense_20_layer_call_and_return_conditional_losses_13593874

inputs1
matmul_readvariableop_resource:	d�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�	
�
)__inference_CRITIC_layer_call_fn_13594083
obs
unknown:	d�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�d
	unknown_4:d
	unknown_5:d
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallobsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_CRITIC_layer_call_and_return_conditional_losses_13593931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������d: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:L H
'
_output_shapes
:���������d

_user_specified_nameobs
�

�
F__inference_dense_21_layer_call_and_return_conditional_losses_13594158

inputs2
matmul_readvariableop_resource:
��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
TanhTanhBiasAdd:output:0*
T0*(
_output_shapes
:����������X
IdentityIdentityTanh:y:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
D__inference_CRITIC_layer_call_and_return_conditional_losses_13593931
obs$
dense_20_13593875:	d� 
dense_20_13593877:	�%
dense_21_13593892:
�� 
dense_21_13593894:	�$
dense_22_13593909:	�d
dense_22_13593911:d#
dense_23_13593925:d
dense_23_13593927:
identity�� dense_20/StatefulPartitionedCall� dense_21/StatefulPartitionedCall� dense_22/StatefulPartitionedCall� dense_23/StatefulPartitionedCall\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Ly
clip_by_value/MinimumMinimumobs clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 * ���
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d�
 dense_20/StatefulPartitionedCallStatefulPartitionedCallclip_by_value:z:0dense_20_13593875dense_20_13593877*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_13593874�
 dense_21/StatefulPartitionedCallStatefulPartitionedCall)dense_20/StatefulPartitionedCall:output:0dense_21_13593892dense_21_13593894*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_21_layer_call_and_return_conditional_losses_13593891�
 dense_22/StatefulPartitionedCallStatefulPartitionedCall)dense_21/StatefulPartitionedCall:output:0dense_22_13593909dense_22_13593911*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������d*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_22_layer_call_and_return_conditional_losses_13593908�
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_13593925dense_23_13593927*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_23_layer_call_and_return_conditional_losses_13593924x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp!^dense_20/StatefulPartitionedCall!^dense_21/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������d: : : : : : : : 2D
 dense_20/StatefulPartitionedCall dense_20/StatefulPartitionedCall2D
 dense_21/StatefulPartitionedCall dense_21/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:L H
'
_output_shapes
:���������d

_user_specified_nameobs
�	
�
)__inference_CRITIC_layer_call_fn_13593950
input_1
unknown:	d�
	unknown_0:	�
	unknown_1:
��
	unknown_2:	�
	unknown_3:	�d
	unknown_4:d
	unknown_5:d
	unknown_6:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *M
fHRF
D__inference_CRITIC_layer_call_and_return_conditional_losses_13593931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������d: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_1
�,
�
#__inference__wrapped_model_13593852
input_1A
.critic_dense_20_matmul_readvariableop_resource:	d�>
/critic_dense_20_biasadd_readvariableop_resource:	�B
.critic_dense_21_matmul_readvariableop_resource:
��>
/critic_dense_21_biasadd_readvariableop_resource:	�A
.critic_dense_22_matmul_readvariableop_resource:	�d=
/critic_dense_22_biasadd_readvariableop_resource:d@
.critic_dense_23_matmul_readvariableop_resource:d=
/critic_dense_23_biasadd_readvariableop_resource:
identity��&CRITIC/dense_20/BiasAdd/ReadVariableOp�%CRITIC/dense_20/MatMul/ReadVariableOp�&CRITIC/dense_21/BiasAdd/ReadVariableOp�%CRITIC/dense_21/MatMul/ReadVariableOp�&CRITIC/dense_22/BiasAdd/ReadVariableOp�%CRITIC/dense_22/MatMul/ReadVariableOp�&CRITIC/dense_23/BiasAdd/ReadVariableOp�%CRITIC/dense_23/MatMul/ReadVariableOpc
CRITIC/clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��L�
CRITIC/clip_by_value/MinimumMinimuminput_1'CRITIC/clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������d[
CRITIC/clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 * ����
CRITIC/clip_by_valueMaximum CRITIC/clip_by_value/Minimum:z:0CRITIC/clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d�
%CRITIC/dense_20/MatMul/ReadVariableOpReadVariableOp.critic_dense_20_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
CRITIC/dense_20/MatMulMatMulCRITIC/clip_by_value:z:0-CRITIC/dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&CRITIC/dense_20/BiasAdd/ReadVariableOpReadVariableOp/critic_dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
CRITIC/dense_20/BiasAddBiasAdd CRITIC/dense_20/MatMul:product:0.CRITIC/dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
CRITIC/dense_20/TanhTanh CRITIC/dense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%CRITIC/dense_21/MatMul/ReadVariableOpReadVariableOp.critic_dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
CRITIC/dense_21/MatMulMatMulCRITIC/dense_20/Tanh:y:0-CRITIC/dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
&CRITIC/dense_21/BiasAdd/ReadVariableOpReadVariableOp/critic_dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
CRITIC/dense_21/BiasAddBiasAdd CRITIC/dense_21/MatMul:product:0.CRITIC/dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������q
CRITIC/dense_21/TanhTanh CRITIC/dense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%CRITIC/dense_22/MatMul/ReadVariableOpReadVariableOp.critic_dense_22_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
CRITIC/dense_22/MatMulMatMulCRITIC/dense_21/Tanh:y:0-CRITIC/dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
&CRITIC/dense_22/BiasAdd/ReadVariableOpReadVariableOp/critic_dense_22_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
CRITIC/dense_22/BiasAddBiasAdd CRITIC/dense_22/MatMul:product:0.CRITIC/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dp
CRITIC/dense_22/TanhTanh CRITIC/dense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
%CRITIC/dense_23/MatMul/ReadVariableOpReadVariableOp.critic_dense_23_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
CRITIC/dense_23/MatMulMatMulCRITIC/dense_22/Tanh:y:0-CRITIC/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
&CRITIC/dense_23/BiasAdd/ReadVariableOpReadVariableOp/critic_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
CRITIC/dense_23/BiasAddBiasAdd CRITIC/dense_23/MatMul:product:0.CRITIC/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������o
IdentityIdentity CRITIC/dense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp'^CRITIC/dense_20/BiasAdd/ReadVariableOp&^CRITIC/dense_20/MatMul/ReadVariableOp'^CRITIC/dense_21/BiasAdd/ReadVariableOp&^CRITIC/dense_21/MatMul/ReadVariableOp'^CRITIC/dense_22/BiasAdd/ReadVariableOp&^CRITIC/dense_22/MatMul/ReadVariableOp'^CRITIC/dense_23/BiasAdd/ReadVariableOp&^CRITIC/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������d: : : : : : : : 2P
&CRITIC/dense_20/BiasAdd/ReadVariableOp&CRITIC/dense_20/BiasAdd/ReadVariableOp2N
%CRITIC/dense_20/MatMul/ReadVariableOp%CRITIC/dense_20/MatMul/ReadVariableOp2P
&CRITIC/dense_21/BiasAdd/ReadVariableOp&CRITIC/dense_21/BiasAdd/ReadVariableOp2N
%CRITIC/dense_21/MatMul/ReadVariableOp%CRITIC/dense_21/MatMul/ReadVariableOp2P
&CRITIC/dense_22/BiasAdd/ReadVariableOp&CRITIC/dense_22/BiasAdd/ReadVariableOp2N
%CRITIC/dense_22/MatMul/ReadVariableOp%CRITIC/dense_22/MatMul/ReadVariableOp2P
&CRITIC/dense_23/BiasAdd/ReadVariableOp&CRITIC/dense_23/BiasAdd/ReadVariableOp2N
%CRITIC/dense_23/MatMul/ReadVariableOp%CRITIC/dense_23/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������d
!
_user_specified_name	input_1
�'
�
D__inference_CRITIC_layer_call_and_return_conditional_losses_13594118
obs:
'dense_20_matmul_readvariableop_resource:	d�7
(dense_20_biasadd_readvariableop_resource:	�;
'dense_21_matmul_readvariableop_resource:
��7
(dense_21_biasadd_readvariableop_resource:	�:
'dense_22_matmul_readvariableop_resource:	�d6
(dense_22_biasadd_readvariableop_resource:d9
'dense_23_matmul_readvariableop_resource:d6
(dense_23_biasadd_readvariableop_resource:
identity��dense_20/BiasAdd/ReadVariableOp�dense_20/MatMul/ReadVariableOp�dense_21/BiasAdd/ReadVariableOp�dense_21/MatMul/ReadVariableOp�dense_22/BiasAdd/ReadVariableOp�dense_22/MatMul/ReadVariableOp�dense_23/BiasAdd/ReadVariableOp�dense_23/MatMul/ReadVariableOp\
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 * ��Ly
clip_by_value/MinimumMinimumobs clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:���������dT
clip_by_value/yConst*
_output_shapes
: *
dtype0*
valueB
 * ���
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:���������d�
dense_20/MatMul/ReadVariableOpReadVariableOp'dense_20_matmul_readvariableop_resource*
_output_shapes
:	d�*
dtype0�
dense_20/MatMulMatMulclip_by_value:z:0&dense_20/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_20/BiasAdd/ReadVariableOpReadVariableOp(dense_20_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_20/BiasAddBiasAdddense_20/MatMul:product:0'dense_20/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_20/TanhTanhdense_20/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_21/MatMul/ReadVariableOpReadVariableOp'dense_21_matmul_readvariableop_resource* 
_output_shapes
:
��*
dtype0�
dense_21/MatMulMatMuldense_20/Tanh:y:0&dense_21/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
dense_21/BiasAdd/ReadVariableOpReadVariableOp(dense_21_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense_21/BiasAddBiasAdddense_21/MatMul:product:0'dense_21/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������c
dense_21/TanhTanhdense_21/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0�
dense_22/MatMulMatMuldense_21/Tanh:y:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������d�
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype0�
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������db
dense_22/TanhTanhdense_22/BiasAdd:output:0*
T0*'
_output_shapes
:���������d�
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes

:d*
dtype0�
dense_23/MatMulMatMuldense_22/Tanh:y:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:����������
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0�
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������h
IdentityIdentitydense_23/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp ^dense_20/BiasAdd/ReadVariableOp^dense_20/MatMul/ReadVariableOp ^dense_21/BiasAdd/ReadVariableOp^dense_21/MatMul/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:���������d: : : : : : : : 2B
dense_20/BiasAdd/ReadVariableOpdense_20/BiasAdd/ReadVariableOp2@
dense_20/MatMul/ReadVariableOpdense_20/MatMul/ReadVariableOp2B
dense_21/BiasAdd/ReadVariableOpdense_21/BiasAdd/ReadVariableOp2@
dense_21/MatMul/ReadVariableOpdense_21/MatMul/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:L H
'
_output_shapes
:���������d

_user_specified_nameobs
�#
�
$__inference__traced_restore_13594278
file_prefix9
&assignvariableop_critic_dense_4_kernel:	d�5
&assignvariableop_1_critic_dense_4_bias:	�<
(assignvariableop_2_critic_dense_5_kernel:
��5
&assignvariableop_3_critic_dense_5_bias:	�;
(assignvariableop_4_critic_dense_6_kernel:	�d4
&assignvariableop_5_critic_dense_6_bias:d:
(assignvariableop_6_critic_dense_7_kernel:d4
&assignvariableop_7_critic_dense_7_bias:

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp&assignvariableop_critic_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp&assignvariableop_1_critic_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp(assignvariableop_2_critic_dense_5_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp&assignvariableop_3_critic_dense_5_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOp(assignvariableop_4_critic_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOp&assignvariableop_5_critic_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp(assignvariableop_6_critic_dense_7_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp&assignvariableop_7_critic_dense_7_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: U

Identity_9IdentityIdentity_8:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*"
_acd_function_control_output(*
_output_shapes
 "!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
F__inference_dense_23_layer_call_and_return_conditional_losses_13593924

inputs0
matmul_readvariableop_resource:d-
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs
�
�
!__inference__traced_save_13594244
file_prefix4
0savev2_critic_dense_4_kernel_read_readvariableop2
.savev2_critic_dense_4_bias_read_readvariableop4
0savev2_critic_dense_5_kernel_read_readvariableop2
.savev2_critic_dense_5_bias_read_readvariableop4
0savev2_critic_dense_6_kernel_read_readvariableop2
.savev2_critic_dense_6_bias_read_readvariableop4
0savev2_critic_dense_7_kernel_read_readvariableop2
.savev2_critic_dense_7_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_critic_dense_4_kernel_read_readvariableop.savev2_critic_dense_4_bias_read_readvariableop0savev2_critic_dense_5_kernel_read_readvariableop.savev2_critic_dense_5_bias_read_readvariableop0savev2_critic_dense_6_kernel_read_readvariableop.savev2_critic_dense_6_bias_read_readvariableop0savev2_critic_dense_7_kernel_read_readvariableop.savev2_critic_dense_7_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*]
_input_shapesL
J: :	d�:�:
��:�:	�d:d:d:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	d�:!

_output_shapes	
:�:&"
 
_output_shapes
:
��:!

_output_shapes	
:�:%!

_output_shapes
:	�d: 

_output_shapes
:d:$ 

_output_shapes

:d: 

_output_shapes
::	

_output_shapes
: 
�

�
F__inference_dense_22_layer_call_and_return_conditional_losses_13593908

inputs1
matmul_readvariableop_resource:	�d-
biasadd_readvariableop_resource:d
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�d*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:d*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������dP
TanhTanhBiasAdd:output:0*
T0*'
_output_shapes
:���������dW
IdentityIdentityTanh:y:0^NoOp*
T0*'
_output_shapes
:���������dw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
+__inference_dense_20_layer_call_fn_13594127

inputs
unknown:	d�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *O
fJRH
F__inference_dense_20_layer_call_and_return_conditional_losses_13593874p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������d
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������d<
output_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:�f
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

dense1

	dense2


dense3
output_layer
	optimizer
loss

signatures
#_self_saveable_object_factories"
_tf_keras_model
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
non_trainable_variables

layers
metrics
layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
trace_0
trace_12�
)__inference_CRITIC_layer_call_fn_13593950
)__inference_CRITIC_layer_call_fn_13594083�
���
FullArgSpec
args�
jself
jobs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0ztrace_1
�
trace_0
 trace_12�
D__inference_CRITIC_layer_call_and_return_conditional_losses_13594118
D__inference_CRITIC_layer_call_and_return_conditional_losses_13594039�
���
FullArgSpec
args�
jself
jobs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 ztrace_0z trace_1
�B�
#__inference__wrapped_model_13593852input_1"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

kernel
bias
#'_self_saveable_object_factories"
_tf_keras_layer
�
(	variables
)trainable_variables
*regularization_losses
+	keras_api
,__call__
*-&call_and_return_all_conditional_losses

kernel
bias
#._self_saveable_object_factories"
_tf_keras_layer
�
/	variables
0trainable_variables
1regularization_losses
2	keras_api
3__call__
*4&call_and_return_all_conditional_losses

kernel
bias
#5_self_saveable_object_factories"
_tf_keras_layer
�
6	variables
7trainable_variables
8regularization_losses
9	keras_api
:__call__
*;&call_and_return_all_conditional_losses

kernel
bias
#<_self_saveable_object_factories"
_tf_keras_layer
"
	optimizer
 "
trackable_dict_wrapper
,
=serving_default"
signature_map
 "
trackable_dict_wrapper
(:&	d�2CRITIC/dense_4/kernel
": �2CRITIC/dense_4/bias
):'
��2CRITIC/dense_5/kernel
": �2CRITIC/dense_5/bias
(:&	�d2CRITIC/dense_6/kernel
!:d2CRITIC/dense_6/bias
':%d2CRITIC/dense_7/kernel
!:2CRITIC/dense_7/bias
 "
trackable_list_wrapper
<
0
	1

2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
)__inference_CRITIC_layer_call_fn_13593950input_1"�
���
FullArgSpec
args�
jself
jobs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
)__inference_CRITIC_layer_call_fn_13594083obs"�
���
FullArgSpec
args�
jself
jobs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_CRITIC_layer_call_and_return_conditional_losses_13594118obs"�
���
FullArgSpec
args�
jself
jobs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
D__inference_CRITIC_layer_call_and_return_conditional_losses_13594039input_1"�
���
FullArgSpec
args�
jself
jobs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
>non_trainable_variables

?layers
@metrics
Alayer_regularization_losses
Blayer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
�
Ctrace_02�
+__inference_dense_20_layer_call_fn_13594127�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zCtrace_0
�
Dtrace_02�
F__inference_dense_20_layer_call_and_return_conditional_losses_13594138�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zDtrace_0
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Enon_trainable_variables

Flayers
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
(	variables
)trainable_variables
*regularization_losses
,__call__
*-&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
�
Jtrace_02�
+__inference_dense_21_layer_call_fn_13594147�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zJtrace_0
�
Ktrace_02�
F__inference_dense_21_layer_call_and_return_conditional_losses_13594158�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zKtrace_0
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lnon_trainable_variables

Mlayers
Nmetrics
Olayer_regularization_losses
Player_metrics
/	variables
0trainable_variables
1regularization_losses
3__call__
*4&call_and_return_all_conditional_losses
&4"call_and_return_conditional_losses"
_generic_user_object
�
Qtrace_02�
+__inference_dense_22_layer_call_fn_13594167�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zQtrace_0
�
Rtrace_02�
F__inference_dense_22_layer_call_and_return_conditional_losses_13594178�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zRtrace_0
 "
trackable_dict_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
Snon_trainable_variables

Tlayers
Umetrics
Vlayer_regularization_losses
Wlayer_metrics
6	variables
7trainable_variables
8regularization_losses
:__call__
*;&call_and_return_all_conditional_losses
&;"call_and_return_conditional_losses"
_generic_user_object
�
Xtrace_02�
+__inference_dense_23_layer_call_fn_13594187�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zXtrace_0
�
Ytrace_02�
F__inference_dense_23_layer_call_and_return_conditional_losses_13594197�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 zYtrace_0
 "
trackable_dict_wrapper
�B�
&__inference_signature_wrapper_13594062input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_20_layer_call_fn_13594127inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_20_layer_call_and_return_conditional_losses_13594138inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_21_layer_call_fn_13594147inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_21_layer_call_and_return_conditional_losses_13594158inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_22_layer_call_fn_13594167inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_22_layer_call_and_return_conditional_losses_13594178inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
+__inference_dense_23_layer_call_fn_13594187inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_dense_23_layer_call_and_return_conditional_losses_13594197inputs"�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
D__inference_CRITIC_layer_call_and_return_conditional_losses_13594039c0�-
&�#
!�
input_1���������d
� "%�"
�
0���������
� �
D__inference_CRITIC_layer_call_and_return_conditional_losses_13594118_,�)
"�
�
obs���������d
� "%�"
�
0���������
� �
)__inference_CRITIC_layer_call_fn_13593950V0�-
&�#
!�
input_1���������d
� "����������
)__inference_CRITIC_layer_call_fn_13594083R,�)
"�
�
obs���������d
� "�����������
#__inference__wrapped_model_13593852q0�-
&�#
!�
input_1���������d
� "3�0
.
output_1"�
output_1����������
F__inference_dense_20_layer_call_and_return_conditional_losses_13594138]/�,
%�"
 �
inputs���������d
� "&�#
�
0����������
� 
+__inference_dense_20_layer_call_fn_13594127P/�,
%�"
 �
inputs���������d
� "������������
F__inference_dense_21_layer_call_and_return_conditional_losses_13594158^0�-
&�#
!�
inputs����������
� "&�#
�
0����������
� �
+__inference_dense_21_layer_call_fn_13594147Q0�-
&�#
!�
inputs����������
� "������������
F__inference_dense_22_layer_call_and_return_conditional_losses_13594178]0�-
&�#
!�
inputs����������
� "%�"
�
0���������d
� 
+__inference_dense_22_layer_call_fn_13594167P0�-
&�#
!�
inputs����������
� "����������d�
F__inference_dense_23_layer_call_and_return_conditional_losses_13594197\/�,
%�"
 �
inputs���������d
� "%�"
�
0���������
� ~
+__inference_dense_23_layer_call_fn_13594187O/�,
%�"
 �
inputs���������d
� "�����������
&__inference_signature_wrapper_13594062|;�8
� 
1�.
,
input_1!�
input_1���������d"3�0
.
output_1"�
output_1���������