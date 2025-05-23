��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
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
A
BroadcastArgs
s0"T
s1"T
r0"T"
Ttype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
�
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
�
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%��8"&
exponential_avg_factorfloat%  �?";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
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
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8��
�
,VARIATIONAL_AUTOENCODER/ENCODER/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:_*=
shared_name.,VARIATIONAL_AUTOENCODER/ENCODER/dense_2/bias
�
@VARIATIONAL_AUTOENCODER/ENCODER/dense_2/bias/Read/ReadVariableOpReadVariableOp,VARIATIONAL_AUTOENCODER/ENCODER/dense_2/bias*
_output_shapes
:_*
dtype0
�
.VARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�_*?
shared_name0.VARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernel
�
BVARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernel/Read/ReadVariableOpReadVariableOp.VARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernel*
_output_shapes
:	�_*
dtype0
�
,VARIATIONAL_AUTOENCODER/ENCODER/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:_*=
shared_name.,VARIATIONAL_AUTOENCODER/ENCODER/dense_1/bias
�
@VARIATIONAL_AUTOENCODER/ENCODER/dense_1/bias/Read/ReadVariableOpReadVariableOp,VARIATIONAL_AUTOENCODER/ENCODER/dense_1/bias*
_output_shapes
:_*
dtype0
�
.VARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	�_*?
shared_name0.VARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernel
�
BVARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernel/Read/ReadVariableOpReadVariableOp.VARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernel*
_output_shapes
:	�_*
dtype0
�
*VARIATIONAL_AUTOENCODER/ENCODER/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*;
shared_name,*VARIATIONAL_AUTOENCODER/ENCODER/dense/bias
�
>VARIATIONAL_AUTOENCODER/ENCODER/dense/bias/Read/ReadVariableOpReadVariableOp*VARIATIONAL_AUTOENCODER/ENCODER/dense/bias*
_output_shapes	
:�*
dtype0
�
,VARIATIONAL_AUTOENCODER/ENCODER/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
�d�*=
shared_name.,VARIATIONAL_AUTOENCODER/ENCODER/dense/kernel
�
@VARIATIONAL_AUTOENCODER/ENCODER/dense/kernel/Read/ReadVariableOpReadVariableOp,VARIATIONAL_AUTOENCODER/ENCODER/dense/kernel* 
_output_shapes
:
�d�*
dtype0
�
-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/bias
�
AVARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/bias/Read/ReadVariableOpReadVariableOp-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/bias*
_output_shapes	
:�*
dtype0
�
/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:��*@
shared_name1/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernel
�
CVARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernel/Read/ReadVariableOpReadVariableOp/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernel*(
_output_shapes
:��*
dtype0
�
-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:�*>
shared_name/-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/bias
�
AVARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/bias/Read/ReadVariableOpReadVariableOp-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/bias*
_output_shapes	
:�*
dtype0
�
/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@�*@
shared_name1/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernel
�
CVARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernel/Read/ReadVariableOpReadVariableOp/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernel*'
_output_shapes
:@�*
dtype0
�
CVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*T
shared_nameECVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_variance
�
WVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOpCVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_variance*
_output_shapes
:@*
dtype0
�
?VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*P
shared_nameA?VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_mean
�
SVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOp?VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_mean*
_output_shapes
:@*
dtype0
�
8VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*I
shared_name:8VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/beta
�
LVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/beta/Read/ReadVariableOpReadVariableOp8VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/beta*
_output_shapes
:@*
dtype0
�
9VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*J
shared_name;9VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gamma
�
MVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gamma/Read/ReadVariableOpReadVariableOp9VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gamma*
_output_shapes
:@*
dtype0
�
-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/bias
�
AVARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/bias/Read/ReadVariableOpReadVariableOp-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/bias*
_output_shapes
:@*
dtype0
�
/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*@
shared_name1/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernel
�
CVARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
�
+VARIATIONAL_AUTOENCODER/ENCODER/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+VARIATIONAL_AUTOENCODER/ENCODER/conv2d/bias
�
?VARIATIONAL_AUTOENCODER/ENCODER/conv2d/bias/Read/ReadVariableOpReadVariableOp+VARIATIONAL_AUTOENCODER/ENCODER/conv2d/bias*
_output_shapes
: *
dtype0
�
-VARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-VARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernel
�
AVARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernel/Read/ReadVariableOpReadVariableOp-VARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernel*&
_output_shapes
: *
dtype0
�
serving_default_input_1Placeholder*0
_output_shapes
:����������P*
dtype0*%
shape:����������P
�	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1-VARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernel+VARIATIONAL_AUTOENCODER/ENCODER/conv2d/bias/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernel-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/bias9VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gamma8VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/beta?VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_meanCVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_variance/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernel-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/bias/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernel-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/bias,VARIATIONAL_AUTOENCODER/ENCODER/dense/kernel*VARIATIONAL_AUTOENCODER/ENCODER/dense/bias.VARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernel,VARIATIONAL_AUTOENCODER/ENCODER/dense_1/bias.VARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernel,VARIATIONAL_AUTOENCODER/ENCODER/dense_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_16589

NoOpNoOp
�B
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�B
value�BB�B B�B
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2

bn1
	conv3
	conv4
flatten

dense1
mu
	sigma
N

signatures*
�
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17*
z
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15*
* 
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
*trace_0
+trace_1
,trace_2
-trace_3* 
6
.trace_0
/trace_1
0trace_2
1trace_3* 
* 
�
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

kernel
bias
 8_jit_compiled_convolution_op*
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias
 ?_jit_compiled_convolution_op*
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
Faxis
	gamma
beta
moving_mean
moving_variance*
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

kernel
bias
 M_jit_compiled_convolution_op*
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

kernel
bias
 T_jit_compiled_convolution_op*
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses* 
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

kernel
 bias*
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

!kernel
"bias*
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

#kernel
$bias*

m_graph_parents* 

nserving_default* 
mg
VARIABLE_VALUE-VARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE+VARIATIONAL_AUTOENCODER/ENCODER/conv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUE9VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gamma&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
xr
VARIABLE_VALUE8VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/beta&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE?VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_mean&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
�}
VARIABLE_VALUECVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_variance&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUE/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUE-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,VARIATIONAL_AUTOENCODER/ENCODER/dense/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*VARIATIONAL_AUTOENCODER/ENCODER/dense/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.VARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,VARIATIONAL_AUTOENCODER/ENCODER/dense_1/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.VARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,VARIATIONAL_AUTOENCODER/ENCODER/dense_2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*

0
1*
C
0
	1

2
3
4
5
6
7
8*
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

0
1*

0
1*
* 
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses*

ttrace_0* 

utrace_0* 
* 

0
1*

0
1*
* 
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses*

{trace_0* 

|trace_0* 
* 
 
0
1
2
3*

0
1*
* 
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 

0
1*

0
1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

0
 1*

0
 1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

!0
"1*

!0
"1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 

#0
$1*

#0
$1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
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

0
1*
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
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAVARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernel/Read/ReadVariableOp?VARIATIONAL_AUTOENCODER/ENCODER/conv2d/bias/Read/ReadVariableOpCVARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernel/Read/ReadVariableOpAVARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/bias/Read/ReadVariableOpMVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gamma/Read/ReadVariableOpLVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/beta/Read/ReadVariableOpSVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_mean/Read/ReadVariableOpWVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_variance/Read/ReadVariableOpCVARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernel/Read/ReadVariableOpAVARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/bias/Read/ReadVariableOpCVARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernel/Read/ReadVariableOpAVARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/bias/Read/ReadVariableOp@VARIATIONAL_AUTOENCODER/ENCODER/dense/kernel/Read/ReadVariableOp>VARIATIONAL_AUTOENCODER/ENCODER/dense/bias/Read/ReadVariableOpBVARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernel/Read/ReadVariableOp@VARIATIONAL_AUTOENCODER/ENCODER/dense_1/bias/Read/ReadVariableOpBVARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernel/Read/ReadVariableOp@VARIATIONAL_AUTOENCODER/ENCODER/dense_2/bias/Read/ReadVariableOpConst*
Tin
2*
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
GPU 2J 8� *'
f"R 
__inference__traced_save_17199
�	
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename-VARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernel+VARIATIONAL_AUTOENCODER/ENCODER/conv2d/bias/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernel-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/bias9VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gamma8VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/beta?VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_meanCVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_variance/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernel-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/bias/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernel-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/bias,VARIATIONAL_AUTOENCODER/ENCODER/dense/kernel*VARIATIONAL_AUTOENCODER/ENCODER/dense/bias.VARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernel,VARIATIONAL_AUTOENCODER/ENCODER/dense_1/bias.VARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernel,VARIATIONAL_AUTOENCODER/ENCODER/dense_2/bias*
Tin
2*
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
!__inference__traced_restore_17263��

�
^
B__inference_flatten_layer_call_and_return_conditional_losses_15897

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������dY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
(__inference_conv2d_1_layer_call_fn_16940

inputs!
unknown: @
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_15842w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������(@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������P( : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������P( 
 
_user_specified_nameinputs
�
C
'__inference_flatten_layer_call_fn_17058

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_15897a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:����������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_15910

inputs2
matmul_readvariableop_resource:
�d�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�d�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_15796

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_17033

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������
�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������
�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������(@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������(@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15885

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������
�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������
�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������
�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�`
�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16546
input_1&
conv2d_16447: 
conv2d_16449: (
conv2d_1_16452: @
conv2d_1_16454:@'
batch_normalization_16457:@'
batch_normalization_16459:@'
batch_normalization_16461:@'
batch_normalization_16463:@)
conv2d_2_16466:@�
conv2d_2_16468:	�*
conv2d_3_16471:��
conv2d_3_16473:	�
dense_16477:
�d�
dense_16479:	� 
dense_1_16482:	�_
dense_1_16484:_ 
dense_2_16487:	�_
dense_2_16489:_
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_16447conv2d_16449*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_15825�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_16452conv2d_1_16454*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_15842�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_16457batch_normalization_16459batch_normalization_16461batch_normalization_16463*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_15796�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_2_16466conv2d_2_16468*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15868�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_16471conv2d_3_16473*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15885�
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_15897�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_16477dense_16479*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15910�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_16482dense_1_16484*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15926�
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_16487dense_2_16489*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15942f
ExpExp(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������_O

Normal/locConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
Normal/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:]
Normal/sample/ConstConst*
_output_shapes
:*
dtype0*
valueB: i
Normal/sample/ProdProdShape:output:0Normal/sample/Const:output:0*
T0*
_output_shapes
: `
Normal/sample/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskb
Normal/sample/shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_2Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
: �
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
: p
Normal/sample/concat/values_0PackNormal/sample/Prod:output:0*
N*
T0*
_output_shapes
:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:���������*
dtype0�
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:����������
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:����������
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/scale:output:0*
T0*#
_output_shapes
:���������t
Normal/sample/addAddV2Normal/sample/mul:z:0Normal/loc:output:0*
T0*#
_output_shapes
:���������X
Normal/sample/ShapeShapeNormal/sample/add:z:0*
T0*
_output_shapes
:m
#Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_2StridedSliceNormal/sample/Shape:output:0,Normal/sample/strided_slice_2/stack:output:0.Normal/sample/strided_slice_2/stack_1:output:0.Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask]
Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concat_1ConcatV2Shape:output:0&Normal/sample/strided_slice_2:output:0$Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Normal/sample/ReshapeReshapeNormal/sample/add:z:0Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:���������_e
mulMulExp:y:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:���������_q
addAddV2(dense_1/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:���������_J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @U
powPowExp:y:0pow/y:output:0*
T0*'
_output_shapes
:���������_L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
pow_1Pow(dense_1/StatefulPartitionedCall:output:0pow_1/y:output:0*
T0*'
_output_shapes
:���������_T
add_1AddV2pow:z:0	pow_1:z:0*
T0*'
_output_shapes
:���������_E
LogLogExp:y:0*
T0*'
_output_shapes
:���������_P
subSub	add_1:z:0Log:y:0*
T0*'
_output_shapes
:���������_L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
sub_1Subsub:z:0sub_1/y:output:0*
T0*'
_output_shapes
:���������_V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	sub_1:z:0Const:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������_�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_1
�
�
'__inference_ENCODER_layer_call_fn_16342
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�d�

unknown_12:	�

unknown_13:	�_

unknown_14:_

unknown_15:	�_

unknown_16:_
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_ENCODER_layer_call_and_return_conditional_losses_16262o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_1
�`
�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16262

inputs&
conv2d_16163: 
conv2d_16165: (
conv2d_1_16168: @
conv2d_1_16170:@'
batch_normalization_16173:@'
batch_normalization_16175:@'
batch_normalization_16177:@'
batch_normalization_16179:@)
conv2d_2_16182:@�
conv2d_2_16184:	�*
conv2d_3_16187:��
conv2d_3_16189:	�
dense_16193:
�d�
dense_16195:	� 
dense_1_16198:	�_
dense_1_16200:_ 
dense_2_16203:	�_
dense_2_16205:_
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_16163conv2d_16165*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_15825�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_16168conv2d_1_16170*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_15842�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_16173batch_normalization_16175batch_normalization_16177batch_normalization_16179*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_15796�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_2_16182conv2d_2_16184*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15868�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_16187conv2d_3_16189*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15885�
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_15897�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_16193dense_16195*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15910�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_16198dense_1_16200*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15926�
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_16203dense_2_16205*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15942f
ExpExp(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������_O

Normal/locConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
Normal/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:]
Normal/sample/ConstConst*
_output_shapes
:*
dtype0*
valueB: i
Normal/sample/ProdProdShape:output:0Normal/sample/Const:output:0*
T0*
_output_shapes
: `
Normal/sample/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskb
Normal/sample/shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_2Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
: �
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
: p
Normal/sample/concat/values_0PackNormal/sample/Prod:output:0*
N*
T0*
_output_shapes
:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:���������*
dtype0�
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:����������
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:����������
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/scale:output:0*
T0*#
_output_shapes
:���������t
Normal/sample/addAddV2Normal/sample/mul:z:0Normal/loc:output:0*
T0*#
_output_shapes
:���������X
Normal/sample/ShapeShapeNormal/sample/add:z:0*
T0*
_output_shapes
:m
#Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_2StridedSliceNormal/sample/Shape:output:0,Normal/sample/strided_slice_2/stack:output:0.Normal/sample/strided_slice_2/stack_1:output:0.Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask]
Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concat_1ConcatV2Shape:output:0&Normal/sample/strided_slice_2:output:0$Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Normal/sample/ReshapeReshapeNormal/sample/add:z:0Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:���������_e
mulMulExp:y:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:���������_q
addAddV2(dense_1/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:���������_J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @U
powPowExp:y:0pow/y:output:0*
T0*'
_output_shapes
:���������_L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
pow_1Pow(dense_1/StatefulPartitionedCall:output:0pow_1/y:output:0*
T0*'
_output_shapes
:���������_T
add_1AddV2pow:z:0	pow_1:z:0*
T0*'
_output_shapes
:���������_E
LogLogExp:y:0*
T0*'
_output_shapes
:���������_P
subSub	add_1:z:0Log:y:0*
T0*'
_output_shapes
:���������_L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
sub_1Subsub:z:0sub_1/y:output:0*
T0*'
_output_shapes
:���������_V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	sub_1:z:0Const:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������_�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_16951

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������(@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������(@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������P( : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������P( 
 
_user_specified_nameinputs
�`
�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16444
input_1&
conv2d_16345: 
conv2d_16347: (
conv2d_1_16350: @
conv2d_1_16352:@'
batch_normalization_16355:@'
batch_normalization_16357:@'
batch_normalization_16359:@'
batch_normalization_16361:@)
conv2d_2_16364:@�
conv2d_2_16366:	�*
conv2d_3_16369:��
conv2d_3_16371:	�
dense_16375:
�d�
dense_16377:	� 
dense_1_16380:	�_
dense_1_16382:_ 
dense_2_16385:	�_
dense_2_16387:_
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_16345conv2d_16347*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_15825�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_16350conv2d_1_16352*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_15842�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_16355batch_normalization_16357batch_normalization_16359batch_normalization_16361*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_15765�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_2_16364conv2d_2_16366*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15868�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_16369conv2d_3_16371*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15885�
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_15897�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_16375dense_16377*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15910�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_16380dense_1_16382*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15926�
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_16385dense_2_16387*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15942f
ExpExp(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������_O

Normal/locConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
Normal/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:]
Normal/sample/ConstConst*
_output_shapes
:*
dtype0*
valueB: i
Normal/sample/ProdProdShape:output:0Normal/sample/Const:output:0*
T0*
_output_shapes
: `
Normal/sample/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskb
Normal/sample/shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_2Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
: �
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
: p
Normal/sample/concat/values_0PackNormal/sample/Prod:output:0*
N*
T0*
_output_shapes
:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:���������*
dtype0�
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:����������
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:����������
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/scale:output:0*
T0*#
_output_shapes
:���������t
Normal/sample/addAddV2Normal/sample/mul:z:0Normal/loc:output:0*
T0*#
_output_shapes
:���������X
Normal/sample/ShapeShapeNormal/sample/add:z:0*
T0*
_output_shapes
:m
#Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_2StridedSliceNormal/sample/Shape:output:0,Normal/sample/strided_slice_2/stack:output:0.Normal/sample/strided_slice_2/stack_1:output:0.Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask]
Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concat_1ConcatV2Shape:output:0&Normal/sample/strided_slice_2:output:0$Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Normal/sample/ReshapeReshapeNormal/sample/add:z:0Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:���������_e
mulMulExp:y:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:���������_q
addAddV2(dense_1/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:���������_J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @U
powPowExp:y:0pow/y:output:0*
T0*'
_output_shapes
:���������_L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
pow_1Pow(dense_1/StatefulPartitionedCall:output:0pow_1/y:output:0*
T0*'
_output_shapes
:���������_T
add_1AddV2pow:z:0	pow_1:z:0*
T0*'
_output_shapes
:���������_E
LogLogExp:y:0*
T0*'
_output_shapes
:���������_P
subSub	add_1:z:0Log:y:0*
T0*'
_output_shapes
:���������_L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
sub_1Subsub:z:0sub_1/y:output:0*
T0*'
_output_shapes
:���������_V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	sub_1:z:0Const:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������_�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_1
�
�
'__inference_ENCODER_layer_call_fn_16671

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�d�

unknown_12:	�

unknown_13:	�_

unknown_14:_

unknown_15:	�_

unknown_16:_
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_ENCODER_layer_call_and_return_conditional_losses_16262o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
��
�
 __inference__wrapped_model_15743
input_1G
-encoder_conv2d_conv2d_readvariableop_resource: <
.encoder_conv2d_biasadd_readvariableop_resource: I
/encoder_conv2d_1_conv2d_readvariableop_resource: @>
0encoder_conv2d_1_biasadd_readvariableop_resource:@A
3encoder_batch_normalization_readvariableop_resource:@C
5encoder_batch_normalization_readvariableop_1_resource:@R
Dencoder_batch_normalization_fusedbatchnormv3_readvariableop_resource:@T
Fencoder_batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@J
/encoder_conv2d_2_conv2d_readvariableop_resource:@�?
0encoder_conv2d_2_biasadd_readvariableop_resource:	�K
/encoder_conv2d_3_conv2d_readvariableop_resource:��?
0encoder_conv2d_3_biasadd_readvariableop_resource:	�@
,encoder_dense_matmul_readvariableop_resource:
�d�<
-encoder_dense_biasadd_readvariableop_resource:	�A
.encoder_dense_1_matmul_readvariableop_resource:	�_=
/encoder_dense_1_biasadd_readvariableop_resource:_A
.encoder_dense_2_matmul_readvariableop_resource:	�_=
/encoder_dense_2_biasadd_readvariableop_resource:_
identity��;ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp�=ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�*ENCODER/batch_normalization/ReadVariableOp�,ENCODER/batch_normalization/ReadVariableOp_1�%ENCODER/conv2d/BiasAdd/ReadVariableOp�$ENCODER/conv2d/Conv2D/ReadVariableOp�'ENCODER/conv2d_1/BiasAdd/ReadVariableOp�&ENCODER/conv2d_1/Conv2D/ReadVariableOp�'ENCODER/conv2d_2/BiasAdd/ReadVariableOp�&ENCODER/conv2d_2/Conv2D/ReadVariableOp�'ENCODER/conv2d_3/BiasAdd/ReadVariableOp�&ENCODER/conv2d_3/Conv2D/ReadVariableOp�$ENCODER/dense/BiasAdd/ReadVariableOp�#ENCODER/dense/MatMul/ReadVariableOp�&ENCODER/dense_1/BiasAdd/ReadVariableOp�%ENCODER/dense_1/MatMul/ReadVariableOp�&ENCODER/dense_2/BiasAdd/ReadVariableOp�%ENCODER/dense_2/MatMul/ReadVariableOp�
$ENCODER/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
ENCODER/conv2d/Conv2DConv2Dinput_1,ENCODER/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( *
paddingSAME*
strides
�
%ENCODER/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
ENCODER/conv2d/BiasAddBiasAddENCODER/conv2d/Conv2D:output:0-ENCODER/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( v
ENCODER/conv2d/ReluReluENCODER/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������P( �
&ENCODER/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
ENCODER/conv2d_1/Conv2DConv2D!ENCODER/conv2d/Relu:activations:0.ENCODER/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@*
paddingSAME*
strides
�
'ENCODER/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
ENCODER/conv2d_1/BiasAddBiasAdd ENCODER/conv2d_1/Conv2D:output:0/ENCODER/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@z
ENCODER/conv2d_1/ReluRelu!ENCODER/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������(@�
*ENCODER/batch_normalization/ReadVariableOpReadVariableOp3encoder_batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
,ENCODER/batch_normalization/ReadVariableOp_1ReadVariableOp5encoder_batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
;ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpDencoder_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
=ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpFencoder_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
,ENCODER/batch_normalization/FusedBatchNormV3FusedBatchNormV3#ENCODER/conv2d_1/Relu:activations:02ENCODER/batch_normalization/ReadVariableOp:value:04ENCODER/batch_normalization/ReadVariableOp_1:value:0CENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0EENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������(@:@:@:@:@:*
epsilon%o�:*
is_training( �
&ENCODER/conv2d_2/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
ENCODER/conv2d_2/Conv2DConv2D0ENCODER/batch_normalization/FusedBatchNormV3:y:0.ENCODER/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
�
'ENCODER/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ENCODER/conv2d_2/BiasAddBiasAdd ENCODER/conv2d_2/Conv2D:output:0/ENCODER/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�{
ENCODER/conv2d_2/ReluRelu!ENCODER/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������
��
&ENCODER/conv2d_3/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
ENCODER/conv2d_3/Conv2DConv2D#ENCODER/conv2d_2/Relu:activations:0.ENCODER/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
�
'ENCODER/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ENCODER/conv2d_3/BiasAddBiasAdd ENCODER/conv2d_3/Conv2D:output:0/ENCODER/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�{
ENCODER/conv2d_3/ReluRelu!ENCODER/conv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:���������
�f
ENCODER/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  �
ENCODER/flatten/ReshapeReshape#ENCODER/conv2d_3/Relu:activations:0ENCODER/flatten/Const:output:0*
T0*(
_output_shapes
:����������d�
#ENCODER/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource* 
_output_shapes
:
�d�*
dtype0�
ENCODER/dense/MatMulMatMul ENCODER/flatten/Reshape:output:0+ENCODER/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:�����������
$ENCODER/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
ENCODER/dense/BiasAddBiasAddENCODER/dense/MatMul:product:0,ENCODER/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������m
ENCODER/dense/ReluReluENCODER/dense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
%ENCODER/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0�
ENCODER/dense_1/MatMulMatMul ENCODER/dense/Relu:activations:0-ENCODER/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
&ENCODER/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0�
ENCODER/dense_1/BiasAddBiasAdd ENCODER/dense_1/MatMul:product:0.ENCODER/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
%ENCODER/dense_2/MatMul/ReadVariableOpReadVariableOp.encoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0�
ENCODER/dense_2/MatMulMatMul ENCODER/dense/Relu:activations:0-ENCODER/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
&ENCODER/dense_2/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_2_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0�
ENCODER/dense_2/BiasAddBiasAdd ENCODER/dense_2/MatMul:product:0.ENCODER/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_f
ENCODER/ExpExp ENCODER/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������_W
ENCODER/Normal/locConst*
_output_shapes
: *
dtype0*
valueB
 *    Y
ENCODER/Normal/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
ENCODER/ShapeShape ENCODER/dense_1/BiasAdd:output:0*
T0*
_output_shapes
:e
ENCODER/Normal/sample/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
ENCODER/Normal/sample/ProdProdENCODER/Shape:output:0$ENCODER/Normal/sample/Const:output:0*
T0*
_output_shapes
: h
%ENCODER/Normal/sample/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB _
ENCODER/Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : s
)ENCODER/Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: u
+ENCODER/Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: u
+ENCODER/Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
#ENCODER/Normal/sample/strided_sliceStridedSlice.ENCODER/Normal/sample/shape_as_tensor:output:02ENCODER/Normal/sample/strided_slice/stack:output:04ENCODER/Normal/sample/strided_slice/stack_1:output:04ENCODER/Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskj
'ENCODER/Normal/sample/shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB _
ENCODER/Normal/sample/Const_2Const*
_output_shapes
: *
dtype0*
value	B : u
+ENCODER/Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: w
-ENCODER/Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-ENCODER/Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%ENCODER/Normal/sample/strided_slice_1StridedSlice0ENCODER/Normal/sample/shape_as_tensor_1:output:04ENCODER/Normal/sample/strided_slice_1/stack:output:06ENCODER/Normal/sample/strided_slice_1/stack_1:output:06ENCODER/Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maski
&ENCODER/Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB k
(ENCODER/Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
#ENCODER/Normal/sample/BroadcastArgsBroadcastArgs1ENCODER/Normal/sample/BroadcastArgs/s0_1:output:0,ENCODER/Normal/sample/strided_slice:output:0*
_output_shapes
: �
%ENCODER/Normal/sample/BroadcastArgs_1BroadcastArgs(ENCODER/Normal/sample/BroadcastArgs:r0:0.ENCODER/Normal/sample/strided_slice_1:output:0*
_output_shapes
: �
%ENCODER/Normal/sample/concat/values_0Pack#ENCODER/Normal/sample/Prod:output:0*
N*
T0*
_output_shapes
:c
!ENCODER/Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
ENCODER/Normal/sample/concatConcatV2.ENCODER/Normal/sample/concat/values_0:output:0*ENCODER/Normal/sample/BroadcastArgs_1:r0:0*ENCODER/Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:t
/ENCODER/Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    v
1ENCODER/Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
?ENCODER/Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormal%ENCODER/Normal/sample/concat:output:0*
T0*#
_output_shapes
:���������*
dtype0�
.ENCODER/Normal/sample/normal/random_normal/mulMulHENCODER/Normal/sample/normal/random_normal/RandomStandardNormal:output:0:ENCODER/Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:����������
*ENCODER/Normal/sample/normal/random_normalAddV22ENCODER/Normal/sample/normal/random_normal/mul:z:08ENCODER/Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:����������
ENCODER/Normal/sample/mulMul.ENCODER/Normal/sample/normal/random_normal:z:0ENCODER/Normal/scale:output:0*
T0*#
_output_shapes
:����������
ENCODER/Normal/sample/addAddV2ENCODER/Normal/sample/mul:z:0ENCODER/Normal/loc:output:0*
T0*#
_output_shapes
:���������h
ENCODER/Normal/sample/ShapeShapeENCODER/Normal/sample/add:z:0*
T0*
_output_shapes
:u
+ENCODER/Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:w
-ENCODER/Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: w
-ENCODER/Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
%ENCODER/Normal/sample/strided_slice_2StridedSlice$ENCODER/Normal/sample/Shape:output:04ENCODER/Normal/sample/strided_slice_2/stack:output:06ENCODER/Normal/sample/strided_slice_2/stack_1:output:06ENCODER/Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_maske
#ENCODER/Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
ENCODER/Normal/sample/concat_1ConcatV2ENCODER/Shape:output:0.ENCODER/Normal/sample/strided_slice_2:output:0,ENCODER/Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
ENCODER/Normal/sample/ReshapeReshapeENCODER/Normal/sample/add:z:0'ENCODER/Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:���������_}
ENCODER/mulMulENCODER/Exp:y:0&ENCODER/Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:���������_y
ENCODER/addAddV2 ENCODER/dense_1/BiasAdd:output:0ENCODER/mul:z:0*
T0*'
_output_shapes
:���������_R
ENCODER/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @m
ENCODER/powPowENCODER/Exp:y:0ENCODER/pow/y:output:0*
T0*'
_output_shapes
:���������_T
ENCODER/pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @�
ENCODER/pow_1Pow ENCODER/dense_1/BiasAdd:output:0ENCODER/pow_1/y:output:0*
T0*'
_output_shapes
:���������_l
ENCODER/add_1AddV2ENCODER/pow:z:0ENCODER/pow_1:z:0*
T0*'
_output_shapes
:���������_U
ENCODER/LogLogENCODER/Exp:y:0*
T0*'
_output_shapes
:���������_h
ENCODER/subSubENCODER/add_1:z:0ENCODER/Log:y:0*
T0*'
_output_shapes
:���������_T
ENCODER/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?q
ENCODER/sub_1SubENCODER/sub:z:0ENCODER/sub_1/y:output:0*
T0*'
_output_shapes
:���������_^
ENCODER/ConstConst*
_output_shapes
:*
dtype0*
valueB"       ^
ENCODER/SumSumENCODER/sub_1:z:0ENCODER/Const:output:0*
T0*
_output_shapes
: ^
IdentityIdentityENCODER/add:z:0^NoOp*
T0*'
_output_shapes
:���������_�
NoOpNoOp<^ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp>^ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp_1+^ENCODER/batch_normalization/ReadVariableOp-^ENCODER/batch_normalization/ReadVariableOp_1&^ENCODER/conv2d/BiasAdd/ReadVariableOp%^ENCODER/conv2d/Conv2D/ReadVariableOp(^ENCODER/conv2d_1/BiasAdd/ReadVariableOp'^ENCODER/conv2d_1/Conv2D/ReadVariableOp(^ENCODER/conv2d_2/BiasAdd/ReadVariableOp'^ENCODER/conv2d_2/Conv2D/ReadVariableOp(^ENCODER/conv2d_3/BiasAdd/ReadVariableOp'^ENCODER/conv2d_3/Conv2D/ReadVariableOp%^ENCODER/dense/BiasAdd/ReadVariableOp$^ENCODER/dense/MatMul/ReadVariableOp'^ENCODER/dense_1/BiasAdd/ReadVariableOp&^ENCODER/dense_1/MatMul/ReadVariableOp'^ENCODER/dense_2/BiasAdd/ReadVariableOp&^ENCODER/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 2z
;ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp;ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp2~
=ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp_1=ENCODER/batch_normalization/FusedBatchNormV3/ReadVariableOp_12X
*ENCODER/batch_normalization/ReadVariableOp*ENCODER/batch_normalization/ReadVariableOp2\
,ENCODER/batch_normalization/ReadVariableOp_1,ENCODER/batch_normalization/ReadVariableOp_12N
%ENCODER/conv2d/BiasAdd/ReadVariableOp%ENCODER/conv2d/BiasAdd/ReadVariableOp2L
$ENCODER/conv2d/Conv2D/ReadVariableOp$ENCODER/conv2d/Conv2D/ReadVariableOp2R
'ENCODER/conv2d_1/BiasAdd/ReadVariableOp'ENCODER/conv2d_1/BiasAdd/ReadVariableOp2P
&ENCODER/conv2d_1/Conv2D/ReadVariableOp&ENCODER/conv2d_1/Conv2D/ReadVariableOp2R
'ENCODER/conv2d_2/BiasAdd/ReadVariableOp'ENCODER/conv2d_2/BiasAdd/ReadVariableOp2P
&ENCODER/conv2d_2/Conv2D/ReadVariableOp&ENCODER/conv2d_2/Conv2D/ReadVariableOp2R
'ENCODER/conv2d_3/BiasAdd/ReadVariableOp'ENCODER/conv2d_3/BiasAdd/ReadVariableOp2P
&ENCODER/conv2d_3/Conv2D/ReadVariableOp&ENCODER/conv2d_3/Conv2D/ReadVariableOp2L
$ENCODER/dense/BiasAdd/ReadVariableOp$ENCODER/dense/BiasAdd/ReadVariableOp2J
#ENCODER/dense/MatMul/ReadVariableOp#ENCODER/dense/MatMul/ReadVariableOp2P
&ENCODER/dense_1/BiasAdd/ReadVariableOp&ENCODER/dense_1/BiasAdd/ReadVariableOp2N
%ENCODER/dense_1/MatMul/ReadVariableOp%ENCODER/dense_1/MatMul/ReadVariableOp2P
&ENCODER/dense_2/BiasAdd/ReadVariableOp&ENCODER/dense_2/BiasAdd/ReadVariableOp2N
%ENCODER/dense_2/MatMul/ReadVariableOp%ENCODER/dense_2/MatMul/ReadVariableOp:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_1
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_16931

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������P( i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������P( w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
��
�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16791

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_2_conv2d_readvariableop_resource:@�7
(conv2d_2_biasadd_readvariableop_resource:	�C
'conv2d_3_conv2d_readvariableop_resource:��7
(conv2d_3_biasadd_readvariableop_resource:	�8
$dense_matmul_readvariableop_resource:
�d�4
%dense_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�_5
'dense_1_biasadd_readvariableop_resource:_9
&dense_2_matmul_readvariableop_resource:	�_5
'dense_2_biasadd_readvariableop_resource:_
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( *
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������P( �
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������(@�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������(@:@:@:@:@:*
epsilon%o�:*
is_training( �
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_2/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������
��
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�k
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:���������
�^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  �
flatten/ReshapeReshapeconv2d_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:����������d�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
�d�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0�
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_V
ExpExpdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������_O

Normal/locConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
Normal/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:]
Normal/sample/ConstConst*
_output_shapes
:*
dtype0*
valueB: i
Normal/sample/ProdProdShape:output:0Normal/sample/Const:output:0*
T0*
_output_shapes
: `
Normal/sample/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskb
Normal/sample/shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_2Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
: �
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
: p
Normal/sample/concat/values_0PackNormal/sample/Prod:output:0*
N*
T0*
_output_shapes
:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:���������*
dtype0�
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:����������
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:����������
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/scale:output:0*
T0*#
_output_shapes
:���������t
Normal/sample/addAddV2Normal/sample/mul:z:0Normal/loc:output:0*
T0*#
_output_shapes
:���������X
Normal/sample/ShapeShapeNormal/sample/add:z:0*
T0*
_output_shapes
:m
#Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_2StridedSliceNormal/sample/Shape:output:0,Normal/sample/strided_slice_2/stack:output:0.Normal/sample/strided_slice_2/stack_1:output:0.Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask]
Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concat_1ConcatV2Shape:output:0&Normal/sample/strided_slice_2:output:0$Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Normal/sample/ReshapeReshapeNormal/sample/add:z:0Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:���������_e
mulMulExp:y:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:���������_a
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:���������_J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @U
powPowExp:y:0pow/y:output:0*
T0*'
_output_shapes
:���������_L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @j
pow_1Powdense_1/BiasAdd:output:0pow_1/y:output:0*
T0*'
_output_shapes
:���������_T
add_1AddV2pow:z:0	pow_1:z:0*
T0*'
_output_shapes
:���������_E
LogLogExp:y:0*
T0*'
_output_shapes
:���������_P
subSub	add_1:z:0Log:y:0*
T0*'
_output_shapes
:���������_L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
sub_1Subsub:z:0sub_1/y:output:0*
T0*'
_output_shapes
:���������_V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	sub_1:z:0Const:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������_�
NoOpNoOp4^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�
�
(__inference_conv2d_3_layer_call_fn_17042

inputs#
unknown:��
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15885x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������
�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������
�: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�
�
(__inference_conv2d_2_layer_call_fn_17022

inputs"
unknown:@�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15868x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:���������
�`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������(@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:���������(@
 
_user_specified_nameinputs
�
�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_15842

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������(@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������(@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������P( : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������P( 
 
_user_specified_nameinputs
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_17103

inputs1
matmul_readvariableop_resource:	�_-
biasadd_readvariableop_resource:_
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������__
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������_w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
'__inference_dense_2_layer_call_fn_17112

inputs
unknown:	�_
	unknown_0:_
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15942o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
A__inference_conv2d_layer_call_and_return_conditional_losses_15825

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( *
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:���������P( i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:���������P( w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������P: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�
�
&__inference_conv2d_layer_call_fn_16920

inputs!
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_15825w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:���������P( `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :����������P: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_15942

inputs1
matmul_readvariableop_resource:	�_-
biasadd_readvariableop_resource:_
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������__
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������_w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17013

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��AssignNewValue�AssignNewValue_1�FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
%__inference_dense_layer_call_fn_17073

inputs
unknown:
�d�
	unknown_0:	�
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15910p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:����������`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������d: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15868

inputs9
conv2d_readvariableop_resource:@�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������
�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������
�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������(@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:���������(@
 
_user_specified_nameinputs
�
�
'__inference_dense_1_layer_call_fn_17093

inputs
unknown:	�_
	unknown_0:_
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15926o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�	
�
B__inference_dense_2_layer_call_and_return_conditional_losses_17122

inputs1
matmul_readvariableop_resource:	�_-
biasadd_readvariableop_resource:_
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������__
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������_w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
�
^
B__inference_flatten_layer_call_and_return_conditional_losses_17064

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:����������dY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:����������d"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:���������
�:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�

�
@__inference_dense_layer_call_and_return_conditional_losses_17084

inputs2
matmul_readvariableop_resource:
�d�.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
�d�*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:����������b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:����������w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������d
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_16589
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�d�

unknown_12:	�

unknown_13:	�_

unknown_14:_

unknown_15:	�_

unknown_16:_
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_15743o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_1
�
�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_17053

inputs:
conv2d_readvariableop_resource:��.
biasadd_readvariableop_resource:	�
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:���������
�j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:���������
�w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :���������
�: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:���������
�
 
_user_specified_nameinputs
�	
�
3__inference_batch_normalization_layer_call_fn_16977

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_15796�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
'__inference_ENCODER_layer_call_fn_16630

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�d�

unknown_12:	�

unknown_13:	�_

unknown_14:_

unknown_15:	�_

unknown_16:_
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_ENCODER_layer_call_and_return_conditional_losses_16002o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�
�
'__inference_ENCODER_layer_call_fn_16041
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@�
	unknown_8:	�%
	unknown_9:��

unknown_10:	�

unknown_11:
�d�

unknown_12:	�

unknown_13:	�_

unknown_14:_

unknown_15:	�_

unknown_16:_
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*4
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_ENCODER_layer_call_and_return_conditional_losses_16002o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������_`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
0
_output_shapes
:����������P
!
_user_specified_name	input_1
�`
�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16002

inputs&
conv2d_15826: 
conv2d_15828: (
conv2d_1_15843: @
conv2d_1_15845:@'
batch_normalization_15848:@'
batch_normalization_15850:@'
batch_normalization_15852:@'
batch_normalization_15854:@)
conv2d_2_15869:@�
conv2d_2_15871:	�*
conv2d_3_15886:��
conv2d_3_15888:	�
dense_15911:
�d�
dense_15913:	� 
dense_1_15927:	�_
dense_1_15929:_ 
dense_2_15943:	�_
dense_2_15945:_
identity��+batch_normalization/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall� conv2d_2/StatefulPartitionedCall� conv2d_3/StatefulPartitionedCall�dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_15826conv2d_15828*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������P( *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_15825�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_15843conv2d_1_15845*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_15842�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0batch_normalization_15848batch_normalization_15850batch_normalization_15852batch_normalization_15854*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:���������(@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_15765�
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:0conv2d_2_15869conv2d_2_15871*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_2_layer_call_and_return_conditional_losses_15868�
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0conv2d_3_15886conv2d_3_15888*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:���������
�*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_conv2d_3_layer_call_and_return_conditional_losses_15885�
flatten/PartitionedCallPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������d* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_flatten_layer_call_and_return_conditional_losses_15897�
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_15911dense_15913*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:����������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_15910�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_15927dense_1_15929*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_15926�
dense_2/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_2_15943dense_2_15945*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������_*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_15942f
ExpExp(dense_2/StatefulPartitionedCall:output:0*
T0*'
_output_shapes
:���������_O

Normal/locConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
Normal/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  �?]
ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:]
Normal/sample/ConstConst*
_output_shapes
:*
dtype0*
valueB: i
Normal/sample/ProdProdShape:output:0Normal/sample/Const:output:0*
T0*
_output_shapes
: `
Normal/sample/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskb
Normal/sample/shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_2Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
: �
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
: p
Normal/sample/concat/values_0PackNormal/sample/Prod:output:0*
N*
T0*
_output_shapes
:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:���������*
dtype0�
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:����������
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:����������
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/scale:output:0*
T0*#
_output_shapes
:���������t
Normal/sample/addAddV2Normal/sample/mul:z:0Normal/loc:output:0*
T0*#
_output_shapes
:���������X
Normal/sample/ShapeShapeNormal/sample/add:z:0*
T0*
_output_shapes
:m
#Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_2StridedSliceNormal/sample/Shape:output:0,Normal/sample/strided_slice_2/stack:output:0.Normal/sample/strided_slice_2/stack_1:output:0.Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask]
Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concat_1ConcatV2Shape:output:0&Normal/sample/strided_slice_2:output:0$Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Normal/sample/ReshapeReshapeNormal/sample/add:z:0Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:���������_e
mulMulExp:y:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:���������_q
addAddV2(dense_1/StatefulPartitionedCall:output:0mul:z:0*
T0*'
_output_shapes
:���������_J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @U
powPowExp:y:0pow/y:output:0*
T0*'
_output_shapes
:���������_L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @z
pow_1Pow(dense_1/StatefulPartitionedCall:output:0pow_1/y:output:0*
T0*'
_output_shapes
:���������_T
add_1AddV2pow:z:0	pow_1:z:0*
T0*'
_output_shapes
:���������_E
LogLogExp:y:0*
T0*'
_output_shapes
:���������_P
subSub	add_1:z:0Log:y:0*
T0*'
_output_shapes
:���������_L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
sub_1Subsub:z:0sub_1/y:output:0*
T0*'
_output_shapes
:���������_V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	sub_1:z:0Const:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������_�
NoOpNoOp,^batch_normalization/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall!^conv2d_3/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�P
�
!__inference__traced_restore_17263
file_prefixX
>assignvariableop_variational_autoencoder_encoder_conv2d_kernel: L
>assignvariableop_1_variational_autoencoder_encoder_conv2d_bias: \
Bassignvariableop_2_variational_autoencoder_encoder_conv2d_1_kernel: @N
@assignvariableop_3_variational_autoencoder_encoder_conv2d_1_bias:@Z
Lassignvariableop_4_variational_autoencoder_encoder_batch_normalization_gamma:@Y
Kassignvariableop_5_variational_autoencoder_encoder_batch_normalization_beta:@`
Rassignvariableop_6_variational_autoencoder_encoder_batch_normalization_moving_mean:@d
Vassignvariableop_7_variational_autoencoder_encoder_batch_normalization_moving_variance:@]
Bassignvariableop_8_variational_autoencoder_encoder_conv2d_2_kernel:@�O
@assignvariableop_9_variational_autoencoder_encoder_conv2d_2_bias:	�_
Cassignvariableop_10_variational_autoencoder_encoder_conv2d_3_kernel:��P
Aassignvariableop_11_variational_autoencoder_encoder_conv2d_3_bias:	�T
@assignvariableop_12_variational_autoencoder_encoder_dense_kernel:
�d�M
>assignvariableop_13_variational_autoencoder_encoder_dense_bias:	�U
Bassignvariableop_14_variational_autoencoder_encoder_dense_1_kernel:	�_N
@assignvariableop_15_variational_autoencoder_encoder_dense_1_bias:_U
Bassignvariableop_16_variational_autoencoder_encoder_dense_2_kernel:	�_N
@assignvariableop_17_variational_autoencoder_encoder_dense_2_bias:_
identity_19��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp>assignvariableop_variational_autoencoder_encoder_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp>assignvariableop_1_variational_autoencoder_encoder_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOpBassignvariableop_2_variational_autoencoder_encoder_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp@assignvariableop_3_variational_autoencoder_encoder_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpLassignvariableop_4_variational_autoencoder_encoder_batch_normalization_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpKassignvariableop_5_variational_autoencoder_encoder_batch_normalization_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOpRassignvariableop_6_variational_autoencoder_encoder_batch_normalization_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOpVassignvariableop_7_variational_autoencoder_encoder_batch_normalization_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOpBassignvariableop_8_variational_autoencoder_encoder_conv2d_2_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp@assignvariableop_9_variational_autoencoder_encoder_conv2d_2_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOpCassignvariableop_10_variational_autoencoder_encoder_conv2d_3_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOpAassignvariableop_11_variational_autoencoder_encoder_conv2d_3_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp@assignvariableop_12_variational_autoencoder_encoder_dense_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp>assignvariableop_13_variational_autoencoder_encoder_dense_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOpBassignvariableop_14_variational_autoencoder_encoder_dense_1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp@assignvariableop_15_variational_autoencoder_encoder_dense_1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOpBassignvariableop_16_variational_autoencoder_encoder_dense_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp@assignvariableop_17_variational_autoencoder_encoder_dense_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 �
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
B__inference_dense_1_layer_call_and_return_conditional_losses_15926

inputs1
matmul_readvariableop_resource:	�_-
biasadd_readvariableop_resource:_
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:_*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������__
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������_w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:����������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:����������
 
_user_specified_nameinputs
��
�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16911

inputs?
%conv2d_conv2d_readvariableop_resource: 4
&conv2d_biasadd_readvariableop_resource: A
'conv2d_1_conv2d_readvariableop_resource: @6
(conv2d_1_biasadd_readvariableop_resource:@9
+batch_normalization_readvariableop_resource:@;
-batch_normalization_readvariableop_1_resource:@J
<batch_normalization_fusedbatchnormv3_readvariableop_resource:@L
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource:@B
'conv2d_2_conv2d_readvariableop_resource:@�7
(conv2d_2_biasadd_readvariableop_resource:	�C
'conv2d_3_conv2d_readvariableop_resource:��7
(conv2d_3_biasadd_readvariableop_resource:	�8
$dense_matmul_readvariableop_resource:
�d�4
%dense_biasadd_readvariableop_resource:	�9
&dense_1_matmul_readvariableop_resource:	�_5
'dense_1_biasadd_readvariableop_resource:_9
&dense_2_matmul_readvariableop_resource:	�_5
'dense_2_biasadd_readvariableop_resource:_
identity��"batch_normalization/AssignNewValue�$batch_normalization/AssignNewValue_1�3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�conv2d_2/BiasAdd/ReadVariableOp�conv2d_2/Conv2D/ReadVariableOp�conv2d_3/BiasAdd/ReadVariableOp�conv2d_3/Conv2D/ReadVariableOp�dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype0�
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( *
paddingSAME*
strides
�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype0�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������P( f
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������P( �
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype0�
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@*
paddingSAME*
strides
�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������(@j
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������(@�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3conv2d_1/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������(@:@:@:@:@:*
epsilon%o�:*
exponential_avg_factor%
�#<�
"batch_normalization/AssignNewValueAssignVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource1batch_normalization/FusedBatchNormV3:batch_mean:04^batch_normalization/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(�
$batch_normalization/AssignNewValue_1AssignVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource5batch_normalization/FusedBatchNormV3:batch_variance:06^batch_normalization/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(�
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*'
_output_shapes
:@�*
dtype0�
conv2d_2/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
�
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�k
conv2d_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:���������
��
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*(
_output_shapes
:��*
dtype0�
conv2d_3/Conv2DConv2Dconv2d_2/Relu:activations:0&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�*
paddingSAME*
strides
�
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:���������
�k
conv2d_3/ReluReluconv2d_3/BiasAdd:output:0*
T0*0
_output_shapes
:���������
�^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"���� 2  �
flatten/ReshapeReshapeconv2d_3/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:����������d�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
�d�*
dtype0�
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:�*
dtype0�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:����������]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:�����������
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	�_*
dtype0�
dense_2/MatMulMatMuldense/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:_*
dtype0�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_V
ExpExpdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������_O

Normal/locConst*
_output_shapes
: *
dtype0*
valueB
 *    Q
Normal/scaleConst*
_output_shapes
: *
dtype0*
valueB
 *  �?M
ShapeShapedense_1/BiasAdd:output:0*
T0*
_output_shapes
:]
Normal/sample/ConstConst*
_output_shapes
:*
dtype0*
valueB: i
Normal/sample/ProdProdShape:output:0Normal/sample/Const:output:0*
T0*
_output_shapes
: `
Normal/sample/shape_as_tensorConst*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_1Const*
_output_shapes
: *
dtype0*
value	B : k
!Normal/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: m
#Normal/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_sliceStridedSlice&Normal/sample/shape_as_tensor:output:0*Normal/sample/strided_slice/stack:output:0,Normal/sample/strided_slice/stack_1:output:0,Normal/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maskb
Normal/sample/shape_as_tensor_1Const*
_output_shapes
: *
dtype0*
valueB W
Normal/sample/Const_2Const*
_output_shapes
: *
dtype0*
value	B : m
#Normal/sample/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_1StridedSlice(Normal/sample/shape_as_tensor_1:output:0,Normal/sample/strided_slice_1/stack:output:0.Normal/sample/strided_slice_1/stack_1:output:0.Normal/sample/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_maska
Normal/sample/BroadcastArgs/s0Const*
_output_shapes
: *
dtype0*
valueB c
 Normal/sample/BroadcastArgs/s0_1Const*
_output_shapes
: *
dtype0*
valueB �
Normal/sample/BroadcastArgsBroadcastArgs)Normal/sample/BroadcastArgs/s0_1:output:0$Normal/sample/strided_slice:output:0*
_output_shapes
: �
Normal/sample/BroadcastArgs_1BroadcastArgs Normal/sample/BroadcastArgs:r0:0&Normal/sample/strided_slice_1:output:0*
_output_shapes
: p
Normal/sample/concat/values_0PackNormal/sample/Prod:output:0*
N*
T0*
_output_shapes
:[
Normal/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concatConcatV2&Normal/sample/concat/values_0:output:0"Normal/sample/BroadcastArgs_1:r0:0"Normal/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:l
'Normal/sample/normal/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    n
)Normal/sample/normal/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  �?�
7Normal/sample/normal/random_normal/RandomStandardNormalRandomStandardNormalNormal/sample/concat:output:0*
T0*#
_output_shapes
:���������*
dtype0�
&Normal/sample/normal/random_normal/mulMul@Normal/sample/normal/random_normal/RandomStandardNormal:output:02Normal/sample/normal/random_normal/stddev:output:0*
T0*#
_output_shapes
:����������
"Normal/sample/normal/random_normalAddV2*Normal/sample/normal/random_normal/mul:z:00Normal/sample/normal/random_normal/mean:output:0*
T0*#
_output_shapes
:����������
Normal/sample/mulMul&Normal/sample/normal/random_normal:z:0Normal/scale:output:0*
T0*#
_output_shapes
:���������t
Normal/sample/addAddV2Normal/sample/mul:z:0Normal/loc:output:0*
T0*#
_output_shapes
:���������X
Normal/sample/ShapeShapeNormal/sample/add:z:0*
T0*
_output_shapes
:m
#Normal/sample/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:o
%Normal/sample/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: o
%Normal/sample/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
Normal/sample/strided_slice_2StridedSliceNormal/sample/Shape:output:0,Normal/sample/strided_slice_2/stack:output:0.Normal/sample/strided_slice_2/stack_1:output:0.Normal/sample/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask]
Normal/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Normal/sample/concat_1ConcatV2Shape:output:0&Normal/sample/strided_slice_2:output:0$Normal/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
Normal/sample/ReshapeReshapeNormal/sample/add:z:0Normal/sample/concat_1:output:0*
T0*'
_output_shapes
:���������_e
mulMulExp:y:0Normal/sample/Reshape:output:0*
T0*'
_output_shapes
:���������_a
addAddV2dense_1/BiasAdd:output:0mul:z:0*
T0*'
_output_shapes
:���������_J
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @U
powPowExp:y:0pow/y:output:0*
T0*'
_output_shapes
:���������_L
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @j
pow_1Powdense_1/BiasAdd:output:0pow_1/y:output:0*
T0*'
_output_shapes
:���������_T
add_1AddV2pow:z:0	pow_1:z:0*
T0*'
_output_shapes
:���������_E
LogLogExp:y:0*
T0*'
_output_shapes
:���������_P
subSub	add_1:z:0Log:y:0*
T0*'
_output_shapes
:���������_L
sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Y
sub_1Subsub:z:0sub_1/y:output:0*
T0*'
_output_shapes
:���������_V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	sub_1:z:0Const:output:0*
T0*
_output_shapes
: V
IdentityIdentityadd:z:0^NoOp*
T0*'
_output_shapes
:���������_�
NoOpNoOp#^batch_normalization/AssignNewValue%^batch_normalization/AssignNewValue_14^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:����������P: : : : : : : : : : : : : : : : : : 2H
"batch_normalization/AssignNewValue"batch_normalization/AssignNewValue2L
$batch_normalization/AssignNewValue_1$batch_normalization/AssignNewValue_12j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:X T
0
_output_shapes
:����������P
 
_user_specified_nameinputs
�	
�
3__inference_batch_normalization_layer_call_fn_16964

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+���������������������������@*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_15765�
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�4
�
__inference__traced_save_17199
file_prefixL
Hsavev2_variational_autoencoder_encoder_conv2d_kernel_read_readvariableopJ
Fsavev2_variational_autoencoder_encoder_conv2d_bias_read_readvariableopN
Jsavev2_variational_autoencoder_encoder_conv2d_1_kernel_read_readvariableopL
Hsavev2_variational_autoencoder_encoder_conv2d_1_bias_read_readvariableopX
Tsavev2_variational_autoencoder_encoder_batch_normalization_gamma_read_readvariableopW
Ssavev2_variational_autoencoder_encoder_batch_normalization_beta_read_readvariableop^
Zsavev2_variational_autoencoder_encoder_batch_normalization_moving_mean_read_readvariableopb
^savev2_variational_autoencoder_encoder_batch_normalization_moving_variance_read_readvariableopN
Jsavev2_variational_autoencoder_encoder_conv2d_2_kernel_read_readvariableopL
Hsavev2_variational_autoencoder_encoder_conv2d_2_bias_read_readvariableopN
Jsavev2_variational_autoencoder_encoder_conv2d_3_kernel_read_readvariableopL
Hsavev2_variational_autoencoder_encoder_conv2d_3_bias_read_readvariableopK
Gsavev2_variational_autoencoder_encoder_dense_kernel_read_readvariableopI
Esavev2_variational_autoencoder_encoder_dense_bias_read_readvariableopM
Isavev2_variational_autoencoder_encoder_dense_1_kernel_read_readvariableopK
Gsavev2_variational_autoencoder_encoder_dense_1_bias_read_readvariableopM
Isavev2_variational_autoencoder_encoder_dense_2_kernel_read_readvariableopK
Gsavev2_variational_autoencoder_encoder_dense_2_bias_read_readvariableop
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
: �
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Hsavev2_variational_autoencoder_encoder_conv2d_kernel_read_readvariableopFsavev2_variational_autoencoder_encoder_conv2d_bias_read_readvariableopJsavev2_variational_autoencoder_encoder_conv2d_1_kernel_read_readvariableopHsavev2_variational_autoencoder_encoder_conv2d_1_bias_read_readvariableopTsavev2_variational_autoencoder_encoder_batch_normalization_gamma_read_readvariableopSsavev2_variational_autoencoder_encoder_batch_normalization_beta_read_readvariableopZsavev2_variational_autoencoder_encoder_batch_normalization_moving_mean_read_readvariableop^savev2_variational_autoencoder_encoder_batch_normalization_moving_variance_read_readvariableopJsavev2_variational_autoencoder_encoder_conv2d_2_kernel_read_readvariableopHsavev2_variational_autoencoder_encoder_conv2d_2_bias_read_readvariableopJsavev2_variational_autoencoder_encoder_conv2d_3_kernel_read_readvariableopHsavev2_variational_autoencoder_encoder_conv2d_3_bias_read_readvariableopGsavev2_variational_autoencoder_encoder_dense_kernel_read_readvariableopEsavev2_variational_autoencoder_encoder_dense_bias_read_readvariableopIsavev2_variational_autoencoder_encoder_dense_1_kernel_read_readvariableopGsavev2_variational_autoencoder_encoder_dense_1_bias_read_readvariableopIsavev2_variational_autoencoder_encoder_dense_2_kernel_read_readvariableopGsavev2_variational_autoencoder_encoder_dense_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *!
dtypes
2�
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

identity_1Identity_1:output:0*�
_input_shapes�
�: : : : @:@:@:@:@:@:@�:�:��:�:
�d�:�:	�_:_:	�_:_: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-	)
'
_output_shapes
:@�:!


_output_shapes	
:�:.*
(
_output_shapes
:��:!

_output_shapes	
:�:&"
 
_output_shapes
:
�d�:!

_output_shapes	
:�:%!

_output_shapes
:	�_: 

_output_shapes
:_:%!

_output_shapes
:	�_: 

_output_shapes
:_:

_output_shapes
: 
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_15765

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16995

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+���������������������������@�
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+���������������������������@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+���������������������������@
 
_user_specified_nameinputs"�	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
D
input_19
serving_default_input_1:0����������P<
output_10
StatefulPartitionedCall:0���������_tensorflow/serving/predict:��
�
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	conv1
		conv2

bn1
	conv3
	conv4
flatten

dense1
mu
	sigma
N

signatures"
_tf_keras_model
�
0
1
2
3
4
5
6
7
8
9
10
11
12
 13
!14
"15
#16
$17"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
8
9
10
 11
!12
"13
#14
$15"
trackable_list_wrapper
 "
trackable_list_wrapper
�
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
*trace_0
+trace_1
,trace_2
-trace_32�
'__inference_ENCODER_layer_call_fn_16041
'__inference_ENCODER_layer_call_fn_16630
'__inference_ENCODER_layer_call_fn_16671
'__inference_ENCODER_layer_call_fn_16342�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z*trace_0z+trace_1z,trace_2z-trace_3
�
.trace_0
/trace_1
0trace_2
1trace_32�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16791
B__inference_ENCODER_layer_call_and_return_conditional_losses_16911
B__inference_ENCODER_layer_call_and_return_conditional_losses_16444
B__inference_ENCODER_layer_call_and_return_conditional_losses_16546�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 z.trace_0z/trace_1z0trace_2z1trace_3
�B�
 __inference__wrapped_model_15743input_1"�
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
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses

kernel
bias
 8_jit_compiled_convolution_op"
_tf_keras_layer
�
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses

kernel
bias
 ?_jit_compiled_convolution_op"
_tf_keras_layer
�
@	variables
Atrainable_variables
Bregularization_losses
C	keras_api
D__call__
*E&call_and_return_all_conditional_losses
Faxis
	gamma
beta
moving_mean
moving_variance"
_tf_keras_layer
�
G	variables
Htrainable_variables
Iregularization_losses
J	keras_api
K__call__
*L&call_and_return_all_conditional_losses

kernel
bias
 M_jit_compiled_convolution_op"
_tf_keras_layer
�
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
R__call__
*S&call_and_return_all_conditional_losses

kernel
bias
 T_jit_compiled_convolution_op"
_tf_keras_layer
�
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_layer
�
[	variables
\trainable_variables
]regularization_losses
^	keras_api
___call__
*`&call_and_return_all_conditional_losses

kernel
 bias"
_tf_keras_layer
�
a	variables
btrainable_variables
cregularization_losses
d	keras_api
e__call__
*f&call_and_return_all_conditional_losses

!kernel
"bias"
_tf_keras_layer
�
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
2
m_graph_parents"
_generic_user_object
,
nserving_default"
signature_map
G:E 2-VARIATIONAL_AUTOENCODER/ENCODER/conv2d/kernel
9:7 2+VARIATIONAL_AUTOENCODER/ENCODER/conv2d/bias
I:G @2/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/kernel
;:9@2-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_1/bias
G:E@29VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/gamma
F:D@28VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/beta
O:M@ (2?VARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_mean
S:Q@ (2CVARIATIONAL_AUTOENCODER/ENCODER/batch_normalization/moving_variance
J:H@�2/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/kernel
<::�2-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_2/bias
K:I��2/VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/kernel
<::�2-VARIATIONAL_AUTOENCODER/ENCODER/conv2d_3/bias
@:>
�d�2,VARIATIONAL_AUTOENCODER/ENCODER/dense/kernel
9:7�2*VARIATIONAL_AUTOENCODER/ENCODER/dense/bias
A:?	�_2.VARIATIONAL_AUTOENCODER/ENCODER/dense_1/kernel
::8_2,VARIATIONAL_AUTOENCODER/ENCODER/dense_1/bias
A:?	�_2.VARIATIONAL_AUTOENCODER/ENCODER/dense_2/kernel
::8_2,VARIATIONAL_AUTOENCODER/ENCODER/dense_2/bias
.
0
1"
trackable_list_wrapper
_
0
	1

2
3
4
5
6
7
8"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
'__inference_ENCODER_layer_call_fn_16041input_1"�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
'__inference_ENCODER_layer_call_fn_16630inputs"�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
'__inference_ENCODER_layer_call_fn_16671inputs"�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
'__inference_ENCODER_layer_call_fn_16342input_1"�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16791inputs"�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16911inputs"�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16444input_1"�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
�B�
B__inference_ENCODER_layer_call_and_return_conditional_losses_16546input_1"�
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
 

kwonlyargs�

jtraining%
kwonlydefaults�

trainingp 
annotations� *
 
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
onon_trainable_variables

players
qmetrics
rlayer_regularization_losses
slayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
�
ttrace_02�
&__inference_conv2d_layer_call_fn_16920�
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
 zttrace_0
�
utrace_02�
A__inference_conv2d_layer_call_and_return_conditional_losses_16931�
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
 zutrace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
vnon_trainable_variables

wlayers
xmetrics
ylayer_regularization_losses
zlayer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
�
{trace_02�
(__inference_conv2d_1_layer_call_fn_16940�
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
 z{trace_0
�
|trace_02�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_16951�
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
 z|trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
<
0
1
2
3"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
}non_trainable_variables

~layers
metrics
 �layer_regularization_losses
�layer_metrics
@	variables
Atrainable_variables
Bregularization_losses
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
3__inference_batch_normalization_layer_call_fn_16964
3__inference_batch_normalization_layer_call_fn_16977�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16995
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17013�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
G	variables
Htrainable_variables
Iregularization_losses
K__call__
*L&call_and_return_all_conditional_losses
&L"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_2_layer_call_fn_17022�
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
 z�trace_0
�
�trace_02�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_17033�
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
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
N	variables
Otrainable_variables
Pregularization_losses
R__call__
*S&call_and_return_all_conditional_losses
&S"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_conv2d_3_layer_call_fn_17042�
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
 z�trace_0
�
�trace_02�
C__inference_conv2d_3_layer_call_and_return_conditional_losses_17053�
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
 z�trace_0
�2��
���
FullArgSpec'
args�
jself
jinputs
jkernel
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
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_flatten_layer_call_fn_17058�
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
 z�trace_0
�
�trace_02�
B__inference_flatten_layer_call_and_return_conditional_losses_17064�
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
 z�trace_0
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
[	variables
\trainable_variables
]regularization_losses
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
%__inference_dense_layer_call_fn_17073�
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
 z�trace_0
�
�trace_02�
@__inference_dense_layer_call_and_return_conditional_losses_17084�
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
 z�trace_0
.
!0
"1"
trackable_list_wrapper
.
!0
"1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
a	variables
btrainable_variables
cregularization_losses
e__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_1_layer_call_fn_17093�
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
 z�trace_0
�
�trace_02�
B__inference_dense_1_layer_call_and_return_conditional_losses_17103�
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
 z�trace_0
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
'__inference_dense_2_layer_call_fn_17112�
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
 z�trace_0
�
�trace_02�
B__inference_dense_2_layer_call_and_return_conditional_losses_17122�
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
 z�trace_0
 "
trackable_list_wrapper
�B�
#__inference_signature_wrapper_16589input_1"�
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
&__inference_conv2d_layer_call_fn_16920inputs"�
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
A__inference_conv2d_layer_call_and_return_conditional_losses_16931inputs"�
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
(__inference_conv2d_1_layer_call_fn_16940inputs"�
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
C__inference_conv2d_1_layer_call_and_return_conditional_losses_16951inputs"�
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
.
0
1"
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
3__inference_batch_normalization_layer_call_fn_16964inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
3__inference_batch_normalization_layer_call_fn_16977inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16995inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17013inputs"�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

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
(__inference_conv2d_2_layer_call_fn_17022inputs"�
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
C__inference_conv2d_2_layer_call_and_return_conditional_losses_17033inputs"�
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
(__inference_conv2d_3_layer_call_fn_17042inputs"�
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
C__inference_conv2d_3_layer_call_and_return_conditional_losses_17053inputs"�
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
'__inference_flatten_layer_call_fn_17058inputs"�
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
B__inference_flatten_layer_call_and_return_conditional_losses_17064inputs"�
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
%__inference_dense_layer_call_fn_17073inputs"�
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
@__inference_dense_layer_call_and_return_conditional_losses_17084inputs"�
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
'__inference_dense_1_layer_call_fn_17093inputs"�
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
B__inference_dense_1_layer_call_and_return_conditional_losses_17103inputs"�
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
'__inference_dense_2_layer_call_fn_17112inputs"�
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
B__inference_dense_2_layer_call_and_return_conditional_losses_17122inputs"�
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
B__inference_ENCODER_layer_call_and_return_conditional_losses_16444� !"#$I�F
/�,
*�'
input_1����������P
�

trainingp "%�"
�
0���������_
� �
B__inference_ENCODER_layer_call_and_return_conditional_losses_16546� !"#$I�F
/�,
*�'
input_1����������P
�

trainingp"%�"
�
0���������_
� �
B__inference_ENCODER_layer_call_and_return_conditional_losses_16791� !"#$H�E
.�+
)�&
inputs����������P
�

trainingp "%�"
�
0���������_
� �
B__inference_ENCODER_layer_call_and_return_conditional_losses_16911� !"#$H�E
.�+
)�&
inputs����������P
�

trainingp"%�"
�
0���������_
� �
'__inference_ENCODER_layer_call_fn_16041y !"#$I�F
/�,
*�'
input_1����������P
�

trainingp "����������_�
'__inference_ENCODER_layer_call_fn_16342y !"#$I�F
/�,
*�'
input_1����������P
�

trainingp"����������_�
'__inference_ENCODER_layer_call_fn_16630x !"#$H�E
.�+
)�&
inputs����������P
�

trainingp "����������_�
'__inference_ENCODER_layer_call_fn_16671x !"#$H�E
.�+
)�&
inputs����������P
�

trainingp"����������_�
 __inference__wrapped_model_15743� !"#$9�6
/�,
*�'
input_1����������P
� "3�0
.
output_1"�
output_1���������_�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_16995�M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_17013�M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
3__inference_batch_normalization_layer_call_fn_16964�M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
3__inference_batch_normalization_layer_call_fn_16977�M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_16951l7�4
-�*
(�%
inputs���������P( 
� "-�*
#� 
0���������(@
� �
(__inference_conv2d_1_layer_call_fn_16940_7�4
-�*
(�%
inputs���������P( 
� " ����������(@�
C__inference_conv2d_2_layer_call_and_return_conditional_losses_17033m7�4
-�*
(�%
inputs���������(@
� ".�+
$�!
0���������
�
� �
(__inference_conv2d_2_layer_call_fn_17022`7�4
-�*
(�%
inputs���������(@
� "!����������
��
C__inference_conv2d_3_layer_call_and_return_conditional_losses_17053n8�5
.�+
)�&
inputs���������
�
� ".�+
$�!
0���������
�
� �
(__inference_conv2d_3_layer_call_fn_17042a8�5
.�+
)�&
inputs���������
�
� "!����������
��
A__inference_conv2d_layer_call_and_return_conditional_losses_16931m8�5
.�+
)�&
inputs����������P
� "-�*
#� 
0���������P( 
� �
&__inference_conv2d_layer_call_fn_16920`8�5
.�+
)�&
inputs����������P
� " ����������P( �
B__inference_dense_1_layer_call_and_return_conditional_losses_17103]!"0�-
&�#
!�
inputs����������
� "%�"
�
0���������_
� {
'__inference_dense_1_layer_call_fn_17093P!"0�-
&�#
!�
inputs����������
� "����������_�
B__inference_dense_2_layer_call_and_return_conditional_losses_17122]#$0�-
&�#
!�
inputs����������
� "%�"
�
0���������_
� {
'__inference_dense_2_layer_call_fn_17112P#$0�-
&�#
!�
inputs����������
� "����������_�
@__inference_dense_layer_call_and_return_conditional_losses_17084^ 0�-
&�#
!�
inputs����������d
� "&�#
�
0����������
� z
%__inference_dense_layer_call_fn_17073Q 0�-
&�#
!�
inputs����������d
� "������������
B__inference_flatten_layer_call_and_return_conditional_losses_17064b8�5
.�+
)�&
inputs���������
�
� "&�#
�
0����������d
� �
'__inference_flatten_layer_call_fn_17058U8�5
.�+
)�&
inputs���������
�
� "�����������d�
#__inference_signature_wrapper_16589� !"#$D�A
� 
:�7
5
input_1*�'
input_1����������P"3�0
.
output_1"�
output_1���������_