       �K"	  �����Abrain.Event:2�����{     �	*T敳��A"��
l
PlaceholderPlaceholder*
shape:WW*
dtype0*&
_output_shapes
:WW
n
Placeholder_1Placeholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_2Placeholder*
shape:��*
dtype0*(
_output_shapes
:��
n
Placeholder_3Placeholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_4Placeholder*
shape:��*
dtype0*(
_output_shapes
:��
M
is_trainingConst*
dtype0
*
_output_shapes
: *
value	B
 Z 
O
is_training_1Const*
value	B
 Z *
dtype0
*
_output_shapes
: 
�
>siamese/scala1/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala1/conv/weights*%
valueB"         `   *
dtype0*
_output_shapes
:
�
=siamese/scala1/conv/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *    
�
?siamese/scala1/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala1/conv/weights/Initializer/truncated_normal/shape*.
_class$
" loc:@siamese/scala1/conv/weights*
seed2 *
dtype0*&
_output_shapes
:`*

seed *
T0
�
<siamese/scala1/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala1/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`
�
8siamese/scala1/conv/weights/Initializer/truncated_normalAdd<siamese/scala1/conv/weights/Initializer/truncated_normal/mul=siamese/scala1/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`
�
siamese/scala1/conv/weights
VariableV2*
shared_name *.
_class$
" loc:@siamese/scala1/conv/weights*
	container *
shape:`*
dtype0*&
_output_shapes
:`
�
"siamese/scala1/conv/weights/AssignAssignsiamese/scala1/conv/weights8siamese/scala1/conv/weights/Initializer/truncated_normal*
validate_shape(*&
_output_shapes
:`*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights
�
 siamese/scala1/conv/weights/readIdentitysiamese/scala1/conv/weights*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`
�
<siamese/scala1/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala1/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala1/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
_output_shapes
: 
�
6siamese/scala1/conv/weights/Regularizer/l2_regularizerMul<siamese/scala1/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala1/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
_output_shapes
: 
�
,siamese/scala1/conv/biases/Initializer/ConstConst*
dtype0*
_output_shapes
:`*-
_class#
!loc:@siamese/scala1/conv/biases*
valueB`*���=
�
siamese/scala1/conv/biases
VariableV2*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name *-
_class#
!loc:@siamese/scala1/conv/biases
�
!siamese/scala1/conv/biases/AssignAssignsiamese/scala1/conv/biases,siamese/scala1/conv/biases/Initializer/Const*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases
�
siamese/scala1/conv/biases/readIdentitysiamese/scala1/conv/biases*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
_output_shapes
:`
�
siamese/scala1/Conv2DConv2DPlaceholder_3 siamese/scala1/conv/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:;;`*
	dilations

�
siamese/scala1/AddAddsiamese/scala1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:;;`*
T0
�
(siamese/scala1/bn/beta/Initializer/ConstConst*)
_class
loc:@siamese/scala1/bn/beta*
valueB`*    *
dtype0*
_output_shapes
:`
�
siamese/scala1/bn/beta
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *)
_class
loc:@siamese/scala1/bn/beta*
	container *
shape:`
�
siamese/scala1/bn/beta/AssignAssignsiamese/scala1/bn/beta(siamese/scala1/bn/beta/Initializer/Const*
_output_shapes
:`*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(
�
siamese/scala1/bn/beta/readIdentitysiamese/scala1/bn/beta*
_output_shapes
:`*
T0*)
_class
loc:@siamese/scala1/bn/beta
�
)siamese/scala1/bn/gamma/Initializer/ConstConst*
_output_shapes
:`**
_class 
loc:@siamese/scala1/bn/gamma*
valueB`*  �?*
dtype0
�
siamese/scala1/bn/gamma
VariableV2*
dtype0*
_output_shapes
:`*
shared_name **
_class 
loc:@siamese/scala1/bn/gamma*
	container *
shape:`
�
siamese/scala1/bn/gamma/AssignAssignsiamese/scala1/bn/gamma)siamese/scala1/bn/gamma/Initializer/Const*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(*
_output_shapes
:`
�
siamese/scala1/bn/gamma/readIdentitysiamese/scala1/bn/gamma*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
_output_shapes
:`
�
/siamese/scala1/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
siamese/scala1/bn/moving_mean
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape:`
�
$siamese/scala1/bn/moving_mean/AssignAssignsiamese/scala1/bn/moving_mean/siamese/scala1/bn/moving_mean/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
"siamese/scala1/bn/moving_mean/readIdentitysiamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
3siamese/scala1/bn/moving_variance/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*  �?*
dtype0*
_output_shapes
:`
�
!siamese/scala1/bn/moving_variance
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape:`
�
(siamese/scala1/bn/moving_variance/AssignAssign!siamese/scala1/bn/moving_variance3siamese/scala1/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
&siamese/scala1/bn/moving_variance/readIdentity!siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
-siamese/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala1/moments/meanMeansiamese/scala1/Add-siamese/scala1/moments/mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
#siamese/scala1/moments/StopGradientStopGradientsiamese/scala1/moments/mean*
T0*&
_output_shapes
:`
�
(siamese/scala1/moments/SquaredDifferenceSquaredDifferencesiamese/scala1/Add#siamese/scala1/moments/StopGradient*&
_output_shapes
:;;`*
T0
�
1siamese/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala1/moments/varianceMean(siamese/scala1/moments/SquaredDifference1siamese/scala1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
siamese/scala1/moments/SqueezeSqueezesiamese/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
 siamese/scala1/moments/Squeeze_1Squeezesiamese/scala1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
�
$siamese/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
3siamese/scala1/siamese/scala1/bn/moving_mean/biased
VariableV2*
_output_shapes
:`*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape:`*
dtype0
�
:siamese/scala1/siamese/scala1/bn/moving_mean/biased/AssignAssign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
8siamese/scala1/siamese/scala1/bn/moving_mean/biased/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Isiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala1/siamese/scala1/bn/moving_mean/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape: 
�
>siamese/scala1/siamese/scala1/bn/moving_mean/local_step/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepIsiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(
�
<siamese/scala1/siamese/scala1/bn/moving_mean/local_step/readIdentity7siamese/scala1/siamese/scala1/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/readsiamese/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMul@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub$siamese/scala1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
isiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biased@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
Lsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepLsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Asiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x$siamese/scala1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Csiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Csiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/x@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivAsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
siamese/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
&siamese/scala1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*
dtype0*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    
�
7siamese/scala1/siamese/scala1/bn/moving_variance/biased
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape:`
�
>siamese/scala1/siamese/scala1/bn/moving_variance/biased/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zeros*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(
�
<siamese/scala1/siamese/scala1/bn/moving_variance/biased/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Msiamese/scala1/siamese/scala1/bn/moving_variance/local_step/Initializer/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala1/siamese/scala1/bn/moving_variance/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape: 
�
Bsiamese/scala1/siamese/scala1/bn/moving_variance/local_step/AssignAssign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepMsiamese/scala1/siamese/scala1/bn/moving_variance/local_step/Initializer/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
@siamese/scala1/siamese/scala1/bn/moving_variance/local_step/readIdentity;siamese/scala1/siamese/scala1/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read siamese/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub&siamese/scala1/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
ssiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Rsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepRsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Gsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x&siamese/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivGsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
 siamese/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
e
siamese/scala1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
g
siamese/scala1/cond/switch_tIdentitysiamese/scala1/cond/Switch:1*
_output_shapes
: *
T0

e
siamese/scala1/cond/switch_fIdentitysiamese/scala1/cond/Switch*
_output_shapes
: *
T0

W
siamese/scala1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala1/cond/Switch_1Switchsiamese/scala1/moments/Squeezesiamese/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*1
_class'
%#loc:@siamese/scala1/moments/Squeeze
�
siamese/scala1/cond/Switch_2Switch siamese/scala1/moments/Squeeze_1siamese/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*3
_class)
'%loc:@siamese/scala1/moments/Squeeze_1
�
siamese/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese/scala1/cond/MergeMergesiamese/scala1/cond/Switch_3siamese/scala1/cond/Switch_1:1*
N*
_output_shapes

:`: *
T0
�
siamese/scala1/cond/Merge_1Mergesiamese/scala1/cond/Switch_4siamese/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
c
siamese/scala1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese/scala1/batchnorm/addAddsiamese/scala1/cond/Merge_1siamese/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
j
siamese/scala1/batchnorm/RsqrtRsqrtsiamese/scala1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese/scala1/batchnorm/mulMulsiamese/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
�
siamese/scala1/batchnorm/mul_1Mulsiamese/scala1/Addsiamese/scala1/batchnorm/mul*
T0*&
_output_shapes
:;;`
�
siamese/scala1/batchnorm/mul_2Mulsiamese/scala1/cond/Mergesiamese/scala1/batchnorm/mul*
_output_shapes
:`*
T0
�
siamese/scala1/batchnorm/subSubsiamese/scala1/bn/beta/readsiamese/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
siamese/scala1/batchnorm/add_1Addsiamese/scala1/batchnorm/mul_1siamese/scala1/batchnorm/sub*
T0*&
_output_shapes
:;;`
l
siamese/scala1/ReluRelusiamese/scala1/batchnorm/add_1*&
_output_shapes
:;;`*
T0
�
siamese/scala1/poll/MaxPoolMaxPoolsiamese/scala1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:`
�
>siamese/scala2/conv/weights/Initializer/truncated_normal/shapeConst*
_output_shapes
:*.
_class$
" loc:@siamese/scala2/conv/weights*%
valueB"      0      *
dtype0
�
=siamese/scala2/conv/weights/Initializer/truncated_normal/meanConst*.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?siamese/scala2/conv/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *���<*
dtype0
�
Hsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala2/conv/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*'
_output_shapes
:0�*

seed *
T0*.
_class$
" loc:@siamese/scala2/conv/weights
�
<siamese/scala2/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala2/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�
�
8siamese/scala2/conv/weights/Initializer/truncated_normalAdd<siamese/scala2/conv/weights/Initializer/truncated_normal/mul=siamese/scala2/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�
�
siamese/scala2/conv/weights
VariableV2*.
_class$
" loc:@siamese/scala2/conv/weights*
	container *
shape:0�*
dtype0*'
_output_shapes
:0�*
shared_name 
�
"siamese/scala2/conv/weights/AssignAssignsiamese/scala2/conv/weights8siamese/scala2/conv/weights/Initializer/truncated_normal*'
_output_shapes
:0�*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(
�
 siamese/scala2/conv/weights/readIdentitysiamese/scala2/conv/weights*'
_output_shapes
:0�*
T0*.
_class$
" loc:@siamese/scala2/conv/weights
�
<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala2/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala2/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala2/conv/weights/read*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala2/conv/weights
�
6siamese/scala2/conv/weights/Regularizer/l2_regularizerMul<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala2/conv/weights/Regularizer/l2_regularizer/L2Loss*.
_class$
" loc:@siamese/scala2/conv/weights*
_output_shapes
: *
T0
�
,siamese/scala2/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala2/conv/biases*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
siamese/scala2/conv/biases
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala2/conv/biases*
	container 
�
!siamese/scala2/conv/biases/AssignAssignsiamese/scala2/conv/biases,siamese/scala2/conv/biases/Initializer/Const*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
siamese/scala2/conv/biases/readIdentitysiamese/scala2/conv/biases*
_output_shapes	
:�*
T0*-
_class#
!loc:@siamese/scala2/conv/biases
V
siamese/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
`
siamese/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/splitSplitsiamese/scala2/split/split_dimsiamese/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:0:0*
	num_split
X
siamese/scala2/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
b
 siamese/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/split_1Split siamese/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese/scala2/Conv2DConv2Dsiamese/scala2/splitsiamese/scala2/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese/scala2/Conv2D_1Conv2Dsiamese/scala2/split:1siamese/scala2/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
\
siamese/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/concatConcatV2siamese/scala2/Conv2Dsiamese/scala2/Conv2D_1siamese/scala2/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese/scala2/AddAddsiamese/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:�*
T0
�
(siamese/scala2/bn/beta/Initializer/ConstConst*)
_class
loc:@siamese/scala2/bn/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala2/bn/beta
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala2/bn/beta*
	container 
�
siamese/scala2/bn/beta/AssignAssignsiamese/scala2/bn/beta(siamese/scala2/bn/beta/Initializer/Const*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�
�
siamese/scala2/bn/beta/readIdentitysiamese/scala2/bn/beta*
T0*)
_class
loc:@siamese/scala2/bn/beta*
_output_shapes	
:�
�
)siamese/scala2/bn/gamma/Initializer/ConstConst**
_class 
loc:@siamese/scala2/bn/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
siamese/scala2/bn/gamma
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name **
_class 
loc:@siamese/scala2/bn/gamma
�
siamese/scala2/bn/gamma/AssignAssignsiamese/scala2/bn/gamma)siamese/scala2/bn/gamma/Initializer/Const*
use_locking(*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
siamese/scala2/bn/gamma/readIdentitysiamese/scala2/bn/gamma*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
_output_shapes	
:�
�
/siamese/scala2/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala2/bn/moving_mean
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
$siamese/scala2/bn/moving_mean/AssignAssignsiamese/scala2/bn/moving_mean/siamese/scala2/bn/moving_mean/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(
�
"siamese/scala2/bn/moving_mean/readIdentitysiamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
3siamese/scala2/bn/moving_variance/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*  �?
�
!siamese/scala2/bn/moving_variance
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container 
�
(siamese/scala2/bn/moving_variance/AssignAssign!siamese/scala2/bn/moving_variance3siamese/scala2/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
&siamese/scala2/bn/moving_variance/readIdentity!siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2/moments/meanMeansiamese/scala2/Add-siamese/scala2/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
#siamese/scala2/moments/StopGradientStopGradientsiamese/scala2/moments/mean*'
_output_shapes
:�*
T0
�
(siamese/scala2/moments/SquaredDifferenceSquaredDifferencesiamese/scala2/Add#siamese/scala2/moments/StopGradient*
T0*'
_output_shapes
:�
�
1siamese/scala2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2/moments/varianceMean(siamese/scala2/moments/SquaredDifference1siamese/scala2/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
siamese/scala2/moments/SqueezeSqueezesiamese/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
 siamese/scala2/moments/Squeeze_1Squeezesiamese/scala2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
$siamese/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3siamese/scala2/siamese/scala2/bn/moving_mean/biased
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape:�
�
:siamese/scala2/siamese/scala2/bn/moving_mean/biased/AssignAssign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zeros*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Isiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala2/siamese/scala2/bn/moving_mean/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape: 
�
>siamese/scala2/siamese/scala2/bn/moving_mean/local_step/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepIsiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
<siamese/scala2/siamese/scala2/bn/moving_mean/local_step/readIdentity7siamese/scala2/siamese/scala2/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readsiamese/scala2/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMul@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub$siamese/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
isiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biased@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Lsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepLsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Asiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x$siamese/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/x@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivAsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
siamese/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
&siamese/scala2/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7siamese/scala2/siamese/scala2/bn/moving_variance/biased
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container *
shape:�
�
>siamese/scala2/siamese/scala2/bn/moving_variance/biased/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
<siamese/scala2/siamese/scala2/bn/moving_variance/biased/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biased*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Msiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/zerosConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *    
�
;siamese/scala2/siamese/scala2/bn/moving_variance/local_step
VariableV2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name 
�
Bsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/AssignAssign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepMsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
@siamese/scala2/siamese/scala2/bn/moving_variance/local_step/readIdentity;siamese/scala2/siamese/scala2/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read siamese/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub&siamese/scala2/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
ssiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Rsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepRsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Gsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x&siamese/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivGsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
 siamese/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
e
siamese/scala2/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
g
siamese/scala2/cond/switch_tIdentitysiamese/scala2/cond/Switch:1*
_output_shapes
: *
T0

e
siamese/scala2/cond/switch_fIdentitysiamese/scala2/cond/Switch*
T0
*
_output_shapes
: 
W
siamese/scala2/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala2/cond/Switch_1Switchsiamese/scala2/moments/Squeezesiamese/scala2/cond/pred_id*
T0*1
_class'
%#loc:@siamese/scala2/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala2/cond/Switch_2Switch siamese/scala2/moments/Squeeze_1siamese/scala2/cond/pred_id*3
_class)
'%loc:@siamese/scala2/moments/Squeeze_1*"
_output_shapes
:�:�*
T0
�
siamese/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala2/cond/MergeMergesiamese/scala2/cond/Switch_3siamese/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala2/cond/Merge_1Mergesiamese/scala2/cond/Switch_4siamese/scala2/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
c
siamese/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala2/batchnorm/addAddsiamese/scala2/cond/Merge_1siamese/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
k
siamese/scala2/batchnorm/RsqrtRsqrtsiamese/scala2/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese/scala2/batchnorm/mulMulsiamese/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
siamese/scala2/batchnorm/mul_1Mulsiamese/scala2/Addsiamese/scala2/batchnorm/mul*'
_output_shapes
:�*
T0
�
siamese/scala2/batchnorm/mul_2Mulsiamese/scala2/cond/Mergesiamese/scala2/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese/scala2/batchnorm/subSubsiamese/scala2/bn/beta/readsiamese/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
siamese/scala2/batchnorm/add_1Addsiamese/scala2/batchnorm/mul_1siamese/scala2/batchnorm/sub*
T0*'
_output_shapes
:�
m
siamese/scala2/ReluRelusiamese/scala2/batchnorm/add_1*'
_output_shapes
:�*
T0
�
siamese/scala2/poll/MaxPoolMaxPoolsiamese/scala2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�
�
>siamese/scala3/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala3/conv/weights*%
valueB"         �  *
dtype0*
_output_shapes
:
�
=siamese/scala3/conv/weights/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *    
�
?siamese/scala3/conv/weights/Initializer/truncated_normal/stddevConst*
_output_shapes
: *.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *���<*
dtype0
�
Hsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala3/conv/weights/Initializer/truncated_normal/shape*

seed *
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
seed2 *
dtype0*(
_output_shapes
:��
�
<siamese/scala3/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala3/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
8siamese/scala3/conv/weights/Initializer/truncated_normalAdd<siamese/scala3/conv/weights/Initializer/truncated_normal/mul=siamese/scala3/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
siamese/scala3/conv/weights
VariableV2*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala3/conv/weights*
	container *
shape:��*
dtype0
�
"siamese/scala3/conv/weights/AssignAssignsiamese/scala3/conv/weights8siamese/scala3/conv/weights/Initializer/truncated_normal*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
�
 siamese/scala3/conv/weights/readIdentitysiamese/scala3/conv/weights*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
<siamese/scala3/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala3/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala3/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
_output_shapes
: 
�
6siamese/scala3/conv/weights/Regularizer/l2_regularizerMul<siamese/scala3/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala3/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
_output_shapes
: 
�
,siamese/scala3/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala3/conv/biases*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
siamese/scala3/conv/biases
VariableV2*
shared_name *-
_class#
!loc:@siamese/scala3/conv/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
!siamese/scala3/conv/biases/AssignAssignsiamese/scala3/conv/biases,siamese/scala3/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�
�
siamese/scala3/conv/biases/readIdentitysiamese/scala3/conv/biases*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
_output_shapes	
:�
�
siamese/scala3/Conv2DConv2Dsiamese/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:

�*
	dilations
*
T0
�
siamese/scala3/AddAddsiamese/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:

�
�
(siamese/scala3/bn/beta/Initializer/ConstConst*)
_class
loc:@siamese/scala3/bn/beta*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/beta
VariableV2*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala3/bn/beta*
	container *
shape:�*
dtype0
�
siamese/scala3/bn/beta/AssignAssignsiamese/scala3/bn/beta(siamese/scala3/bn/beta/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(
�
siamese/scala3/bn/beta/readIdentitysiamese/scala3/bn/beta*)
_class
loc:@siamese/scala3/bn/beta*
_output_shapes	
:�*
T0
�
)siamese/scala3/bn/gamma/Initializer/ConstConst**
_class 
loc:@siamese/scala3/bn/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/gamma
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name **
_class 
loc:@siamese/scala3/bn/gamma
�
siamese/scala3/bn/gamma/AssignAssignsiamese/scala3/bn/gamma)siamese/scala3/bn/gamma/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma
�
siamese/scala3/bn/gamma/readIdentitysiamese/scala3/bn/gamma*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
_output_shapes	
:�
�
/siamese/scala3/bn/moving_mean/Initializer/ConstConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0
�
siamese/scala3/bn/moving_mean
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
$siamese/scala3/bn/moving_mean/AssignAssignsiamese/scala3/bn/moving_mean/siamese/scala3/bn/moving_mean/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
"siamese/scala3/bn/moving_mean/readIdentitysiamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
3siamese/scala3/bn/moving_variance/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
!siamese/scala3/bn/moving_variance
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container 
�
(siamese/scala3/bn/moving_variance/AssignAssign!siamese/scala3/bn/moving_variance3siamese/scala3/bn/moving_variance/Initializer/Const*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
&siamese/scala3/bn/moving_variance/readIdentity!siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala3/moments/meanMeansiamese/scala3/Add-siamese/scala3/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
#siamese/scala3/moments/StopGradientStopGradientsiamese/scala3/moments/mean*
T0*'
_output_shapes
:�
�
(siamese/scala3/moments/SquaredDifferenceSquaredDifferencesiamese/scala3/Add#siamese/scala3/moments/StopGradient*'
_output_shapes
:

�*
T0
�
1siamese/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala3/moments/varianceMean(siamese/scala3/moments/SquaredDifference1siamese/scala3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
siamese/scala3/moments/SqueezeSqueezesiamese/scala3/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
 siamese/scala3/moments/Squeeze_1Squeezesiamese/scala3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
$siamese/scala3/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3siamese/scala3/siamese/scala3/bn/moving_mean/biased
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
:siamese/scala3/siamese/scala3/bn/moving_mean/biased/AssignAssign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zeros*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Isiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala3/siamese/scala3/bn/moving_mean/local_step
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container 
�
>siamese/scala3/siamese/scala3/bn/moving_mean/local_step/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepIsiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/zeros*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
<siamese/scala3/siamese/scala3/bn/moving_mean/local_step/readIdentity7siamese/scala3/siamese/scala3/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readsiamese/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMul@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub$siamese/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
isiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biased@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Lsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepLsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Asiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x$siamese/scala3/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/x@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivAsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
siamese/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
&siamese/scala3/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7siamese/scala3/siamese/scala3/bn/moving_variance/biased
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container 
�
>siamese/scala3/siamese/scala3/bn/moving_variance/biased/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
<siamese/scala3/siamese/scala3/bn/moving_variance/biased/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Msiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala3/siamese/scala3/bn/moving_variance/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape: 
�
Bsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/AssignAssign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepMsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/zeros*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(
�
@siamese/scala3/siamese/scala3/bn/moving_variance/local_step/readIdentity;siamese/scala3/siamese/scala3/bn/moving_variance/local_step*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read siamese/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub&siamese/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
ssiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Rsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepRsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Gsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x&siamese/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Isiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivGsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
 siamese/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
e
siamese/scala3/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

g
siamese/scala3/cond/switch_tIdentitysiamese/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
e
siamese/scala3/cond/switch_fIdentitysiamese/scala3/cond/Switch*
T0
*
_output_shapes
: 
W
siamese/scala3/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala3/cond/Switch_1Switchsiamese/scala3/moments/Squeezesiamese/scala3/cond/pred_id*
T0*1
_class'
%#loc:@siamese/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_2Switch siamese/scala3/moments/Squeeze_1siamese/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala3/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese/scala3/cond/MergeMergesiamese/scala3/cond/Switch_3siamese/scala3/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese/scala3/cond/Merge_1Mergesiamese/scala3/cond/Switch_4siamese/scala3/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
c
siamese/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala3/batchnorm/addAddsiamese/scala3/cond/Merge_1siamese/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
k
siamese/scala3/batchnorm/RsqrtRsqrtsiamese/scala3/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese/scala3/batchnorm/mulMulsiamese/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/mul_1Mulsiamese/scala3/Addsiamese/scala3/batchnorm/mul*
T0*'
_output_shapes
:

�
�
siamese/scala3/batchnorm/mul_2Mulsiamese/scala3/cond/Mergesiamese/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/subSubsiamese/scala3/bn/beta/readsiamese/scala3/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
siamese/scala3/batchnorm/add_1Addsiamese/scala3/batchnorm/mul_1siamese/scala3/batchnorm/sub*
T0*'
_output_shapes
:

�
m
siamese/scala3/ReluRelusiamese/scala3/batchnorm/add_1*'
_output_shapes
:

�*
T0
�
>siamese/scala4/conv/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@siamese/scala4/conv/weights*%
valueB"      �   �  
�
=siamese/scala4/conv/weights/Initializer/truncated_normal/meanConst*.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?siamese/scala4/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala4/conv/weights/Initializer/truncated_normal/shape*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
seed2 *
dtype0*(
_output_shapes
:��*

seed 
�
<siamese/scala4/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala4/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��
�
8siamese/scala4/conv/weights/Initializer/truncated_normalAdd<siamese/scala4/conv/weights/Initializer/truncated_normal/mul=siamese/scala4/conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
�
siamese/scala4/conv/weights
VariableV2*
shape:��*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala4/conv/weights*
	container 
�
"siamese/scala4/conv/weights/AssignAssignsiamese/scala4/conv/weights8siamese/scala4/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��
�
 siamese/scala4/conv/weights/readIdentitysiamese/scala4/conv/weights*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
�
<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala4/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala4/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala4/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
_output_shapes
: 
�
6siamese/scala4/conv/weights/Regularizer/l2_regularizerMul<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala4/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
_output_shapes
: 
�
,siamese/scala4/conv/biases/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*-
_class#
!loc:@siamese/scala4/conv/biases*
valueB�*���=
�
siamese/scala4/conv/biases
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala4/conv/biases
�
!siamese/scala4/conv/biases/AssignAssignsiamese/scala4/conv/biases,siamese/scala4/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(*
_output_shapes	
:�
�
siamese/scala4/conv/biases/readIdentitysiamese/scala4/conv/biases*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
_output_shapes	
:�
V
siamese/scala4/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
`
siamese/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/splitSplitsiamese/scala4/split/split_dimsiamese/scala3/Relu*:
_output_shapes(
&:

�:

�*
	num_split*
T0
X
siamese/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/split_1Split siamese/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese/scala4/Conv2DConv2Dsiamese/scala4/splitsiamese/scala4/split_1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC
�
siamese/scala4/Conv2D_1Conv2Dsiamese/scala4/split:1siamese/scala4/split_1:1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
\
siamese/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/concatConcatV2siamese/scala4/Conv2Dsiamese/scala4/Conv2D_1siamese/scala4/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese/scala4/AddAddsiamese/scala4/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
�
(siamese/scala4/bn/beta/Initializer/ConstConst*
_output_shapes	
:�*)
_class
loc:@siamese/scala4/bn/beta*
valueB�*    *
dtype0
�
siamese/scala4/bn/beta
VariableV2*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala4/bn/beta*
	container *
shape:�*
dtype0
�
siamese/scala4/bn/beta/AssignAssignsiamese/scala4/bn/beta(siamese/scala4/bn/beta/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta
�
siamese/scala4/bn/beta/readIdentitysiamese/scala4/bn/beta*
T0*)
_class
loc:@siamese/scala4/bn/beta*
_output_shapes	
:�
�
)siamese/scala4/bn/gamma/Initializer/ConstConst**
_class 
loc:@siamese/scala4/bn/gamma*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
siamese/scala4/bn/gamma
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name **
_class 
loc:@siamese/scala4/bn/gamma*
	container *
shape:�
�
siamese/scala4/bn/gamma/AssignAssignsiamese/scala4/bn/gamma)siamese/scala4/bn/gamma/Initializer/Const*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
siamese/scala4/bn/gamma/readIdentitysiamese/scala4/bn/gamma*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
_output_shapes	
:�
�
/siamese/scala4/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala4/bn/moving_mean
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape:�
�
$siamese/scala4/bn/moving_mean/AssignAssignsiamese/scala4/bn/moving_mean/siamese/scala4/bn/moving_mean/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
"siamese/scala4/bn/moving_mean/readIdentitysiamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
3siamese/scala4/bn/moving_variance/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
!siamese/scala4/bn/moving_variance
VariableV2*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
(siamese/scala4/bn/moving_variance/AssignAssign!siamese/scala4/bn/moving_variance3siamese/scala4/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
&siamese/scala4/bn/moving_variance/readIdentity!siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala4/moments/meanMeansiamese/scala4/Add-siamese/scala4/moments/mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
#siamese/scala4/moments/StopGradientStopGradientsiamese/scala4/moments/mean*
T0*'
_output_shapes
:�
�
(siamese/scala4/moments/SquaredDifferenceSquaredDifferencesiamese/scala4/Add#siamese/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
1siamese/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala4/moments/varianceMean(siamese/scala4/moments/SquaredDifference1siamese/scala4/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
siamese/scala4/moments/SqueezeSqueezesiamese/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
 siamese/scala4/moments/Squeeze_1Squeezesiamese/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
$siamese/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
3siamese/scala4/siamese/scala4/bn/moving_mean/biased
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape:�
�
:siamese/scala4/siamese/scala4/bn/moving_mean/biased/AssignAssign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zeros*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
8siamese/scala4/siamese/scala4/bn/moving_mean/biased/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Isiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala4/siamese/scala4/bn/moving_mean/local_step
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
>siamese/scala4/siamese/scala4/bn/moving_mean/local_step/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepIsiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/zeros*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
<siamese/scala4/siamese/scala4/bn/moving_mean/local_step/readIdentity7siamese/scala4/siamese/scala4/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/readsiamese/scala4/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMul@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub$siamese/scala4/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
isiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biased@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Lsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Fsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepLsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Asiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x$siamese/scala4/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Csiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Csiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/x@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivAsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
siamese/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
&siamese/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
7siamese/scala4/siamese/scala4/bn/moving_variance/biased
VariableV2*
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container 
�
>siamese/scala4/siamese/scala4/bn/moving_variance/biased/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zeros*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
<siamese/scala4/siamese/scala4/bn/moving_variance/biased/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Msiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala4/siamese/scala4/bn/moving_variance/local_step
VariableV2*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape: *
dtype0
�
Bsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/AssignAssign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepMsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/zeros*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
@siamese/scala4/siamese/scala4/bn/moving_variance/local_step/readIdentity;siamese/scala4/siamese/scala4/bn/moving_variance/local_step*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read siamese/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub&siamese/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
ssiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Rsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Lsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepRsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Gsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x&siamese/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Isiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Isiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivGsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
 siamese/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
e
siamese/scala4/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
g
siamese/scala4/cond/switch_tIdentitysiamese/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
e
siamese/scala4/cond/switch_fIdentitysiamese/scala4/cond/Switch*
T0
*
_output_shapes
: 
W
siamese/scala4/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala4/cond/Switch_1Switchsiamese/scala4/moments/Squeezesiamese/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*1
_class'
%#loc:@siamese/scala4/moments/Squeeze
�
siamese/scala4/cond/Switch_2Switch siamese/scala4/moments/Squeeze_1siamese/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese/scala4/cond/MergeMergesiamese/scala4/cond/Switch_3siamese/scala4/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese/scala4/cond/Merge_1Mergesiamese/scala4/cond/Switch_4siamese/scala4/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
c
siamese/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala4/batchnorm/addAddsiamese/scala4/cond/Merge_1siamese/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
k
siamese/scala4/batchnorm/RsqrtRsqrtsiamese/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala4/batchnorm/mulMulsiamese/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
siamese/scala4/batchnorm/mul_1Mulsiamese/scala4/Addsiamese/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
siamese/scala4/batchnorm/mul_2Mulsiamese/scala4/cond/Mergesiamese/scala4/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese/scala4/batchnorm/subSubsiamese/scala4/bn/beta/readsiamese/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
siamese/scala4/batchnorm/add_1Addsiamese/scala4/batchnorm/mul_1siamese/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
m
siamese/scala4/ReluRelusiamese/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
�
>siamese/scala5/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala5/conv/weights*%
valueB"      �      *
dtype0*
_output_shapes
:
�
=siamese/scala5/conv/weights/Initializer/truncated_normal/meanConst*.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *    *
dtype0*
_output_shapes
: 
�
?siamese/scala5/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala5/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala5/conv/weights/Initializer/truncated_normal/shape*.
_class$
" loc:@siamese/scala5/conv/weights*
seed2 *
dtype0*(
_output_shapes
:��*

seed *
T0
�
<siamese/scala5/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala5/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala5/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*(
_output_shapes
:��
�
8siamese/scala5/conv/weights/Initializer/truncated_normalAdd<siamese/scala5/conv/weights/Initializer/truncated_normal/mul=siamese/scala5/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*(
_output_shapes
:��
�
siamese/scala5/conv/weights
VariableV2*
shared_name *.
_class$
" loc:@siamese/scala5/conv/weights*
	container *
shape:��*
dtype0*(
_output_shapes
:��
�
"siamese/scala5/conv/weights/AssignAssignsiamese/scala5/conv/weights8siamese/scala5/conv/weights/Initializer/truncated_normal*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(
�
 siamese/scala5/conv/weights/readIdentitysiamese/scala5/conv/weights*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*(
_output_shapes
:��
�
<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scaleConst*.
_class$
" loc:@siamese/scala5/conv/weights*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala5/conv/weights/read*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala5/conv/weights
�
6siamese/scala5/conv/weights/Regularizer/l2_regularizerMul<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2Loss*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
_output_shapes
: 
�
,siamese/scala5/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala5/conv/biases*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
siamese/scala5/conv/biases
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala5/conv/biases
�
!siamese/scala5/conv/biases/AssignAssignsiamese/scala5/conv/biases,siamese/scala5/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�
�
siamese/scala5/conv/biases/readIdentitysiamese/scala5/conv/biases*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
_output_shapes	
:�
V
siamese/scala5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
`
siamese/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5/splitSplitsiamese/scala5/split/split_dimsiamese/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
X
siamese/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5/split_1Split siamese/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese/scala5/Conv2DConv2Dsiamese/scala5/splitsiamese/scala5/split_1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

�
siamese/scala5/Conv2D_1Conv2Dsiamese/scala5/split:1siamese/scala5/split_1:1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
\
siamese/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5/concatConcatV2siamese/scala5/Conv2Dsiamese/scala5/Conv2D_1siamese/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese/scala5/AddAddsiamese/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
�
siamese/scala1_1/Conv2DConv2DPlaceholder_4 siamese/scala1/conv/weights/read*
paddingVALID*&
_output_shapes
:{{`*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese/scala1_1/AddAddsiamese/scala1_1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:{{`
�
/siamese/scala1_1/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese/scala1_1/moments/meanMeansiamese/scala1_1/Add/siamese/scala1_1/moments/mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
%siamese/scala1_1/moments/StopGradientStopGradientsiamese/scala1_1/moments/mean*
T0*&
_output_shapes
:`
�
*siamese/scala1_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala1_1/Add%siamese/scala1_1/moments/StopGradient*&
_output_shapes
:{{`*
T0
�
3siamese/scala1_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese/scala1_1/moments/varianceMean*siamese/scala1_1/moments/SquaredDifference3siamese/scala1_1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
 siamese/scala1_1/moments/SqueezeSqueezesiamese/scala1_1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
"siamese/scala1_1/moments/Squeeze_1Squeeze!siamese/scala1_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
�
&siamese/scala1_1/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese/scala1_1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese/scala1_1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
ksiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Nsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese/scala1_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese/scala1_1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
(siamese/scala1_1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese/scala1_1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese/scala1_1/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
usiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese/scala1_1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese/scala1_1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( 
g
siamese/scala1_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala1_1/cond/switch_tIdentitysiamese/scala1_1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese/scala1_1/cond/switch_fIdentitysiamese/scala1_1/cond/Switch*
_output_shapes
: *
T0

Y
siamese/scala1_1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala1_1/cond/Switch_1Switch siamese/scala1_1/moments/Squeezesiamese/scala1_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala1_1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_2Switch"siamese/scala1_1/moments/Squeeze_1siamese/scala1_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala1_1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese/scala1_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese/scala1_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/MergeMergesiamese/scala1_1/cond/Switch_3 siamese/scala1_1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese/scala1_1/cond/Merge_1Mergesiamese/scala1_1/cond/Switch_4 siamese/scala1_1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese/scala1_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese/scala1_1/batchnorm/addAddsiamese/scala1_1/cond/Merge_1 siamese/scala1_1/batchnorm/add/y*
_output_shapes
:`*
T0
n
 siamese/scala1_1/batchnorm/RsqrtRsqrtsiamese/scala1_1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese/scala1_1/batchnorm/mulMul siamese/scala1_1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
T0*
_output_shapes
:`
�
 siamese/scala1_1/batchnorm/mul_1Mulsiamese/scala1_1/Addsiamese/scala1_1/batchnorm/mul*
T0*&
_output_shapes
:{{`
�
 siamese/scala1_1/batchnorm/mul_2Mulsiamese/scala1_1/cond/Mergesiamese/scala1_1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese/scala1_1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese/scala1_1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese/scala1_1/batchnorm/add_1Add siamese/scala1_1/batchnorm/mul_1siamese/scala1_1/batchnorm/sub*&
_output_shapes
:{{`*
T0
p
siamese/scala1_1/ReluRelu siamese/scala1_1/batchnorm/add_1*&
_output_shapes
:{{`*
T0
�
siamese/scala1_1/poll/MaxPoolMaxPoolsiamese/scala1_1/Relu*&
_output_shapes
:==`*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
X
siamese/scala2_1/ConstConst*
_output_shapes
: *
value	B :*
dtype0
b
 siamese/scala2_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/splitSplit siamese/scala2_1/split/split_dimsiamese/scala1_1/poll/MaxPool*8
_output_shapes&
$:==0:==0*
	num_split*
T0
Z
siamese/scala2_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese/scala2_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/split_1Split"siamese/scala2_1/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
�
siamese/scala2_1/Conv2DConv2Dsiamese/scala2_1/splitsiamese/scala2_1/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�
�
siamese/scala2_1/Conv2D_1Conv2Dsiamese/scala2_1/split:1siamese/scala2_1/split_1:1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�
^
siamese/scala2_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/concatConcatV2siamese/scala2_1/Conv2Dsiamese/scala2_1/Conv2D_1siamese/scala2_1/concat/axis*
T0*
N*'
_output_shapes
:99�*

Tidx0
�
siamese/scala2_1/AddAddsiamese/scala2_1/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:99�*
T0
�
/siamese/scala2_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2_1/moments/meanMeansiamese/scala2_1/Add/siamese/scala2_1/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese/scala2_1/moments/StopGradientStopGradientsiamese/scala2_1/moments/mean*
T0*'
_output_shapes
:�
�
*siamese/scala2_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala2_1/Add%siamese/scala2_1/moments/StopGradient*'
_output_shapes
:99�*
T0
�
3siamese/scala2_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese/scala2_1/moments/varianceMean*siamese/scala2_1/moments/SquaredDifference3siamese/scala2_1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese/scala2_1/moments/SqueezeSqueezesiamese/scala2_1/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese/scala2_1/moments/Squeeze_1Squeeze!siamese/scala2_1/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese/scala2_1/AssignMovingAvg/decayConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese/scala2_1/moments/Squeeze*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese/scala2_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Nsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese/scala2_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
 siamese/scala2_1/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala2_1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese/scala2_1/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese/scala2_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Tsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Nsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese/scala2_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
"siamese/scala2_1/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
g
siamese/scala2_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala2_1/cond/switch_tIdentitysiamese/scala2_1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese/scala2_1/cond/switch_fIdentitysiamese/scala2_1/cond/Switch*
_output_shapes
: *
T0

Y
siamese/scala2_1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala2_1/cond/Switch_1Switch siamese/scala2_1/moments/Squeezesiamese/scala2_1/cond/pred_id*3
_class)
'%loc:@siamese/scala2_1/moments/Squeeze*"
_output_shapes
:�:�*
T0
�
siamese/scala2_1/cond/Switch_2Switch"siamese/scala2_1/moments/Squeeze_1siamese/scala2_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala2_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala2_1/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese/scala2_1/cond/pred_id*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese/scala2_1/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese/scala2_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala2_1/cond/MergeMergesiamese/scala2_1/cond/Switch_3 siamese/scala2_1/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese/scala2_1/cond/Merge_1Mergesiamese/scala2_1/cond/Switch_4 siamese/scala2_1/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese/scala2_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/batchnorm/addAddsiamese/scala2_1/cond/Merge_1 siamese/scala2_1/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese/scala2_1/batchnorm/RsqrtRsqrtsiamese/scala2_1/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala2_1/batchnorm/mulMul siamese/scala2_1/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese/scala2_1/batchnorm/mul_1Mulsiamese/scala2_1/Addsiamese/scala2_1/batchnorm/mul*
T0*'
_output_shapes
:99�
�
 siamese/scala2_1/batchnorm/mul_2Mulsiamese/scala2_1/cond/Mergesiamese/scala2_1/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala2_1/batchnorm/subSubsiamese/scala2/bn/beta/read siamese/scala2_1/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese/scala2_1/batchnorm/add_1Add siamese/scala2_1/batchnorm/mul_1siamese/scala2_1/batchnorm/sub*
T0*'
_output_shapes
:99�
q
siamese/scala2_1/ReluRelu siamese/scala2_1/batchnorm/add_1*
T0*'
_output_shapes
:99�
�
siamese/scala2_1/poll/MaxPoolMaxPoolsiamese/scala2_1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�
�
siamese/scala3_1/Conv2DConv2Dsiamese/scala2_1/poll/MaxPool siamese/scala3/conv/weights/read*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
�
siamese/scala3_1/AddAddsiamese/scala3_1/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese/scala3_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala3_1/moments/meanMeansiamese/scala3_1/Add/siamese/scala3_1/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese/scala3_1/moments/StopGradientStopGradientsiamese/scala3_1/moments/mean*'
_output_shapes
:�*
T0
�
*siamese/scala3_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala3_1/Add%siamese/scala3_1/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese/scala3_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese/scala3_1/moments/varianceMean*siamese/scala3_1/moments/SquaredDifference3siamese/scala3_1/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese/scala3_1/moments/SqueezeSqueezesiamese/scala3_1/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese/scala3_1/moments/Squeeze_1Squeeze!siamese/scala3_1/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese/scala3_1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese/scala3_1/moments/Squeeze*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese/scala3_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Nsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?
�
Hsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese/scala3_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
 siamese/scala3_1/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala3_1/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese/scala3_1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese/scala3_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Tsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Isiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese/scala3_1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
"siamese/scala3_1/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
g
siamese/scala3_1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
k
siamese/scala3_1/cond/switch_tIdentitysiamese/scala3_1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese/scala3_1/cond/switch_fIdentitysiamese/scala3_1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese/scala3_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala3_1/cond/Switch_1Switch siamese/scala3_1/moments/Squeezesiamese/scala3_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala3_1/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/Switch_2Switch"siamese/scala3_1/moments/Squeeze_1siamese/scala3_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala3_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese/scala3_1/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
siamese/scala3_1/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese/scala3_1/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese/scala3_1/cond/MergeMergesiamese/scala3_1/cond/Switch_3 siamese/scala3_1/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala3_1/cond/Merge_1Mergesiamese/scala3_1/cond/Switch_4 siamese/scala3_1/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese/scala3_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
�
siamese/scala3_1/batchnorm/addAddsiamese/scala3_1/cond/Merge_1 siamese/scala3_1/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese/scala3_1/batchnorm/RsqrtRsqrtsiamese/scala3_1/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala3_1/batchnorm/mulMul siamese/scala3_1/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese/scala3_1/batchnorm/mul_1Mulsiamese/scala3_1/Addsiamese/scala3_1/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese/scala3_1/batchnorm/mul_2Mulsiamese/scala3_1/cond/Mergesiamese/scala3_1/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese/scala3_1/batchnorm/subSubsiamese/scala3/bn/beta/read siamese/scala3_1/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese/scala3_1/batchnorm/add_1Add siamese/scala3_1/batchnorm/mul_1siamese/scala3_1/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese/scala3_1/ReluRelu siamese/scala3_1/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese/scala4_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese/scala4_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/splitSplit siamese/scala4_1/split/split_dimsiamese/scala3_1/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese/scala4_1/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese/scala4_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/split_1Split"siamese/scala4_1/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese/scala4_1/Conv2DConv2Dsiamese/scala4_1/splitsiamese/scala4_1/split_1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala4_1/Conv2D_1Conv2Dsiamese/scala4_1/split:1siamese/scala4_1/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
^
siamese/scala4_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/concatConcatV2siamese/scala4_1/Conv2Dsiamese/scala4_1/Conv2D_1siamese/scala4_1/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese/scala4_1/AddAddsiamese/scala4_1/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese/scala4_1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala4_1/moments/meanMeansiamese/scala4_1/Add/siamese/scala4_1/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese/scala4_1/moments/StopGradientStopGradientsiamese/scala4_1/moments/mean*
T0*'
_output_shapes
:�
�
*siamese/scala4_1/moments/SquaredDifferenceSquaredDifferencesiamese/scala4_1/Add%siamese/scala4_1/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese/scala4_1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese/scala4_1/moments/varianceMean*siamese/scala4_1/moments/SquaredDifference3siamese/scala4_1/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese/scala4_1/moments/SqueezeSqueezesiamese/scala4_1/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese/scala4_1/moments/Squeeze_1Squeeze!siamese/scala4_1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese/scala4_1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese/scala4_1/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese/scala4_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese/scala4_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
 siamese/scala4_1/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala4_1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese/scala4_1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese/scala4_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese/scala4_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Ksiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
"siamese/scala4_1/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
g
siamese/scala4_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala4_1/cond/switch_tIdentitysiamese/scala4_1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese/scala4_1/cond/switch_fIdentitysiamese/scala4_1/cond/Switch*
_output_shapes
: *
T0

Y
siamese/scala4_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala4_1/cond/Switch_1Switch siamese/scala4_1/moments/Squeezesiamese/scala4_1/cond/pred_id*
T0*3
_class)
'%loc:@siamese/scala4_1/moments/Squeeze*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_2Switch"siamese/scala4_1/moments/Squeeze_1siamese/scala4_1/cond/pred_id*
T0*5
_class+
)'loc:@siamese/scala4_1/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese/scala4_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese/scala4_1/cond/pred_id*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese/scala4_1/cond/MergeMergesiamese/scala4_1/cond/Switch_3 siamese/scala4_1/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala4_1/cond/Merge_1Mergesiamese/scala4_1/cond/Switch_4 siamese/scala4_1/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese/scala4_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/batchnorm/addAddsiamese/scala4_1/cond/Merge_1 siamese/scala4_1/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese/scala4_1/batchnorm/RsqrtRsqrtsiamese/scala4_1/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala4_1/batchnorm/mulMul siamese/scala4_1/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese/scala4_1/batchnorm/mul_1Mulsiamese/scala4_1/Addsiamese/scala4_1/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese/scala4_1/batchnorm/mul_2Mulsiamese/scala4_1/cond/Mergesiamese/scala4_1/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese/scala4_1/batchnorm/subSubsiamese/scala4/bn/beta/read siamese/scala4_1/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese/scala4_1/batchnorm/add_1Add siamese/scala4_1/batchnorm/mul_1siamese/scala4_1/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese/scala4_1/ReluRelu siamese/scala4_1/batchnorm/add_1*'
_output_shapes
:�*
T0
X
siamese/scala5_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese/scala5_1/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese/scala5_1/splitSplit siamese/scala5_1/split/split_dimsiamese/scala4_1/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese/scala5_1/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese/scala5_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/split_1Split"siamese/scala5_1/split_1/split_dim siamese/scala5/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese/scala5_1/Conv2DConv2Dsiamese/scala5_1/splitsiamese/scala5_1/split_1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC
�
siamese/scala5_1/Conv2D_1Conv2Dsiamese/scala5_1/split:1siamese/scala5_1/split_1:1*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
^
siamese/scala5_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/concatConcatV2siamese/scala5_1/Conv2Dsiamese/scala5_1/Conv2D_1siamese/scala5_1/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese/scala5_1/AddAddsiamese/scala5_1/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
m
score/transpose/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
score/transpose	Transposesiamese/scala5/Addscore/transpose/perm*
T0*'
_output_shapes
:�*
Tperm0
M
score/ConstConst*
_output_shapes
: *
value	B :*
dtype0
W
score/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
score/splitSplitscore/split/split_dimscore/transpose*
T0*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split
O
score/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
Y
score/split_1/split_dimConst*
_output_shapes
: *
value	B : *
dtype0
�
score/split_1Splitscore/split_1/split_dimsiamese/scala5_1/Add*
T0*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split
�
score/Conv2DConv2Dscore/split_1score/split*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score/Conv2D_1Conv2Dscore/split_1:1score/split:1*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
score/Conv2D_2Conv2Dscore/split_1:2score/split:2*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations

�
score/Conv2D_3Conv2Dscore/split_1:3score/split:3*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0
�
score/Conv2D_4Conv2Dscore/split_1:4score/split:4*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score/Conv2D_5Conv2Dscore/split_1:5score/split:5*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0
�
score/Conv2D_6Conv2Dscore/split_1:6score/split:6*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
score/Conv2D_7Conv2Dscore/split_1:7score/split:7*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
S
score/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
score/concatConcatV2score/Conv2Dscore/Conv2D_1score/Conv2D_2score/Conv2D_3score/Conv2D_4score/Conv2D_5score/Conv2D_6score/Conv2D_7score/concat/axis*
T0*
N*&
_output_shapes
:*

Tidx0
o
score/transpose_1/permConst*%
valueB"             *
dtype0*
_output_shapes
:
�
score/transpose_1	Transposescore/concatscore/transpose_1/perm*&
_output_shapes
:*
Tperm0*
T0
�
 adjust/weights/Initializer/ConstConst*!
_class
loc:@adjust/weights*%
valueB*o�:*
dtype0*&
_output_shapes
:
�
adjust/weights
VariableV2*
dtype0*&
_output_shapes
:*
shared_name *!
_class
loc:@adjust/weights*
	container *
shape:
�
adjust/weights/AssignAssignadjust/weights adjust/weights/Initializer/Const*
use_locking(*
T0*!
_class
loc:@adjust/weights*
validate_shape(*&
_output_shapes
:
�
adjust/weights/readIdentityadjust/weights*&
_output_shapes
:*
T0*!
_class
loc:@adjust/weights
�
/adjust/weights/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *!
_class
loc:@adjust/weights*
valueB
 *o:
�
0adjust/weights/Regularizer/l2_regularizer/L2LossL2Lossadjust/weights/read*
T0*!
_class
loc:@adjust/weights*
_output_shapes
: 
�
)adjust/weights/Regularizer/l2_regularizerMul/adjust/weights/Regularizer/l2_regularizer/scale0adjust/weights/Regularizer/l2_regularizer/L2Loss*
T0*!
_class
loc:@adjust/weights*
_output_shapes
: 
�
adjust/biases/Initializer/ConstConst* 
_class
loc:@adjust/biases*
valueB*    *
dtype0*
_output_shapes
:
�
adjust/biases
VariableV2*
dtype0*
_output_shapes
:*
shared_name * 
_class
loc:@adjust/biases*
	container *
shape:
�
adjust/biases/AssignAssignadjust/biasesadjust/biases/Initializer/Const*
use_locking(*
T0* 
_class
loc:@adjust/biases*
validate_shape(*
_output_shapes
:
t
adjust/biases/readIdentityadjust/biases*
T0* 
_class
loc:@adjust/biases*
_output_shapes
:
�
.adjust/biases/Regularizer/l2_regularizer/scaleConst* 
_class
loc:@adjust/biases*
valueB
 *o:*
dtype0*
_output_shapes
: 
�
/adjust/biases/Regularizer/l2_regularizer/L2LossL2Lossadjust/biases/read*
_output_shapes
: *
T0* 
_class
loc:@adjust/biases
�
(adjust/biases/Regularizer/l2_regularizerMul.adjust/biases/Regularizer/l2_regularizer/scale/adjust/biases/Regularizer/l2_regularizer/L2Loss*
T0* 
_class
loc:@adjust/biases*
_output_shapes
: 
�
adjust/Conv2DConv2Dscore/transpose_1adjust/weights/read*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
e

adjust/AddAddadjust/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:
P

save/ConstConst*
dtype0*
_output_shapes
: *
valueB Bmodel
�
save/SaveV2/tensor_namesConst*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights*
dtype0*
_output_shapes
:,
�
save/SaveV2/shape_and_slicesConst*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:,
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesadjust/biasesadjust/weightssiamese/scala1/bn/betasiamese/scala1/bn/gammasiamese/scala1/bn/moving_mean!siamese/scala1/bn/moving_variancesiamese/scala1/conv/biasessiamese/scala1/conv/weights3siamese/scala1/siamese/scala1/bn/moving_mean/biased7siamese/scala1/siamese/scala1/bn/moving_mean/local_step7siamese/scala1/siamese/scala1/bn/moving_variance/biased;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsiamese/scala2/bn/betasiamese/scala2/bn/gammasiamese/scala2/bn/moving_mean!siamese/scala2/bn/moving_variancesiamese/scala2/conv/biasessiamese/scala2/conv/weights3siamese/scala2/siamese/scala2/bn/moving_mean/biased7siamese/scala2/siamese/scala2/bn/moving_mean/local_step7siamese/scala2/siamese/scala2/bn/moving_variance/biased;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsiamese/scala3/bn/betasiamese/scala3/bn/gammasiamese/scala3/bn/moving_mean!siamese/scala3/bn/moving_variancesiamese/scala3/conv/biasessiamese/scala3/conv/weights3siamese/scala3/siamese/scala3/bn/moving_mean/biased7siamese/scala3/siamese/scala3/bn/moving_mean/local_step7siamese/scala3/siamese/scala3/bn/moving_variance/biased;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsiamese/scala4/bn/betasiamese/scala4/bn/gammasiamese/scala4/bn/moving_mean!siamese/scala4/bn/moving_variancesiamese/scala4/conv/biasessiamese/scala4/conv/weights3siamese/scala4/siamese/scala4/bn/moving_mean/biased7siamese/scala4/siamese/scala4/bn/moving_mean/local_step7siamese/scala4/siamese/scala4/bn/moving_variance/biased;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsiamese/scala5/conv/biasessiamese/scala5/conv/weights*:
dtypes0
.2,
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
_output_shapes
: *
T0*
_class
loc:@save/Const
�
save/RestoreV2/tensor_namesConst"/device:CPU:0*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights*
dtype0*
_output_shapes
:,
�
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:,
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::*:
dtypes0
.2,
�
save/AssignAssignadjust/biasessave/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@adjust/biases
�
save/Assign_1Assignadjust/weightssave/RestoreV2:1*
T0*!
_class
loc:@adjust/weights*
validate_shape(*&
_output_shapes
:*
use_locking(
�
save/Assign_2Assignsiamese/scala1/bn/betasave/RestoreV2:2*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(*
_output_shapes
:`
�
save/Assign_3Assignsiamese/scala1/bn/gammasave/RestoreV2:3*
_output_shapes
:`*
use_locking(*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(
�
save/Assign_4Assignsiamese/scala1/bn/moving_meansave/RestoreV2:4*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
save/Assign_5Assign!siamese/scala1/bn/moving_variancesave/RestoreV2:5*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
save/Assign_6Assignsiamese/scala1/conv/biasessave/RestoreV2:6*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
�
save/Assign_7Assignsiamese/scala1/conv/weightssave/RestoreV2:7*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`*
use_locking(
�
save/Assign_8Assign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedsave/RestoreV2:8*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
save/Assign_9Assign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepsave/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
save/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave/RestoreV2:10*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(
�
save/Assign_11Assign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsave/RestoreV2:11*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
save/Assign_12Assignsiamese/scala2/bn/betasave/RestoreV2:12*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�
�
save/Assign_13Assignsiamese/scala2/bn/gammasave/RestoreV2:13*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_14Assignsiamese/scala2/bn/moving_meansave/RestoreV2:14*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_15Assign!siamese/scala2/bn/moving_variancesave/RestoreV2:15*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
save/Assign_16Assignsiamese/scala2/conv/biasessave/RestoreV2:16*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_17Assignsiamese/scala2/conv/weightssave/RestoreV2:17*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�
�
save/Assign_18Assign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedsave/RestoreV2:18*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave/RestoreV2:19*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/Assign_20Assign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedsave/RestoreV2:20*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave/RestoreV2:21*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save/Assign_22Assignsiamese/scala3/bn/betasave/RestoreV2:22*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(*
_output_shapes	
:�
�
save/Assign_23Assignsiamese/scala3/bn/gammasave/RestoreV2:23*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
save/Assign_24Assignsiamese/scala3/bn/moving_meansave/RestoreV2:24*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_25Assign!siamese/scala3/bn/moving_variancesave/RestoreV2:25*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
save/Assign_26Assignsiamese/scala3/conv/biasessave/RestoreV2:26*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/Assign_27Assignsiamese/scala3/conv/weightssave/RestoreV2:27*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��
�
save/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave/RestoreV2:28*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_29Assign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepsave/RestoreV2:29*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(
�
save/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave/RestoreV2:30*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave/RestoreV2:31*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_32Assignsiamese/scala4/bn/betasave/RestoreV2:32*
use_locking(*
T0*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(*
_output_shapes	
:�
�
save/Assign_33Assignsiamese/scala4/bn/gammasave/RestoreV2:33*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
save/Assign_34Assignsiamese/scala4/bn/moving_meansave/RestoreV2:34*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_35Assign!siamese/scala4/bn/moving_variancesave/RestoreV2:35*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(
�
save/Assign_36Assignsiamese/scala4/conv/biasessave/RestoreV2:36*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(
�
save/Assign_37Assignsiamese/scala4/conv/weightssave/RestoreV2:37*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��
�
save/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave/RestoreV2:38*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave/RestoreV2:39*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
save/Assign_40Assign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedsave/RestoreV2:40*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
save/Assign_41Assign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsave/RestoreV2:41*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_42Assignsiamese/scala5/conv/biasessave/RestoreV2:42*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save/Assign_43Assignsiamese/scala5/conv/weightssave/RestoreV2:43*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
�
siamese_1/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*&
_output_shapes
:;;`*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_1/scala1/AddAddsiamese_1/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:;;`
�
/siamese_1/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala1/moments/meanMeansiamese_1/scala1/Add/siamese_1/scala1/moments/mean/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
%siamese_1/scala1/moments/StopGradientStopGradientsiamese_1/scala1/moments/mean*
T0*&
_output_shapes
:`
�
*siamese_1/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala1/Add%siamese_1/scala1/moments/StopGradient*
T0*&
_output_shapes
:;;`
�
3siamese_1/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_1/scala1/moments/varianceMean*siamese_1/scala1/moments/SquaredDifference3siamese_1/scala1/moments/variance/reduction_indices*
T0*&
_output_shapes
:`*
	keep_dims(*

Tidx0
�
 siamese_1/scala1/moments/SqueezeSqueezesiamese_1/scala1/moments/mean*
T0*
_output_shapes
:`*
squeeze_dims
 
�
"siamese_1/scala1/moments/Squeeze_1Squeeze!siamese_1/scala1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
�
&siamese_1/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0*
_output_shapes
:`
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_1/scala1/moments/Squeeze*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_1/scala1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
ksiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Nsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_1/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_1/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
(siamese_1/scala1/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_1/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_1/scala1/AssignMovingAvg_1/decay*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
usiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_1/scala1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese_1/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
c
siamese_1/scala1/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_1/scala1/cond/switch_tIdentitysiamese_1/scala1/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_1/scala1/cond/switch_fIdentitysiamese_1/scala1/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala1/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala1/cond/Switch_1Switch siamese_1/scala1/moments/Squeezesiamese_1/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_2Switch"siamese_1/scala1/moments/Squeeze_1siamese_1/scala1/cond/pred_id*
T0*5
_class+
)'loc:@siamese_1/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_1/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_1/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/MergeMergesiamese_1/scala1/cond/Switch_3 siamese_1/scala1/cond/Switch_1:1*
_output_shapes

:`: *
T0*
N
�
siamese_1/scala1/cond/Merge_1Mergesiamese_1/scala1/cond/Switch_4 siamese_1/scala1/cond/Switch_2:1*
_output_shapes

:`: *
T0*
N
e
 siamese_1/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala1/batchnorm/addAddsiamese_1/scala1/cond/Merge_1 siamese_1/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese_1/scala1/batchnorm/RsqrtRsqrtsiamese_1/scala1/batchnorm/add*
_output_shapes
:`*
T0
�
siamese_1/scala1/batchnorm/mulMul siamese_1/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
�
 siamese_1/scala1/batchnorm/mul_1Mulsiamese_1/scala1/Addsiamese_1/scala1/batchnorm/mul*
T0*&
_output_shapes
:;;`
�
 siamese_1/scala1/batchnorm/mul_2Mulsiamese_1/scala1/cond/Mergesiamese_1/scala1/batchnorm/mul*
_output_shapes
:`*
T0
�
siamese_1/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_1/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese_1/scala1/batchnorm/add_1Add siamese_1/scala1/batchnorm/mul_1siamese_1/scala1/batchnorm/sub*&
_output_shapes
:;;`*
T0
p
siamese_1/scala1/ReluRelu siamese_1/scala1/batchnorm/add_1*&
_output_shapes
:;;`*
T0
�
siamese_1/scala1/poll/MaxPoolMaxPoolsiamese_1/scala1/Relu*
ksize
*
paddingVALID*&
_output_shapes
:`*
T0*
data_formatNHWC*
strides

X
siamese_1/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_1/scala2/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_1/scala2/splitSplit siamese_1/scala2/split/split_dimsiamese_1/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:0:0*
	num_split
Z
siamese_1/scala2/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese_1/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/split_1Split"siamese_1/scala2/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
�
siamese_1/scala2/Conv2DConv2Dsiamese_1/scala2/splitsiamese_1/scala2/split_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese_1/scala2/Conv2D_1Conv2Dsiamese_1/scala2/split:1siamese_1/scala2/split_1:1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

^
siamese_1/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/concatConcatV2siamese_1/scala2/Conv2Dsiamese_1/scala2/Conv2D_1siamese_1/scala2/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_1/scala2/AddAddsiamese_1/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_1/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala2/moments/meanMeansiamese_1/scala2/Add/siamese_1/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_1/scala2/moments/StopGradientStopGradientsiamese_1/scala2/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_1/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala2/Add%siamese_1/scala2/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_1/scala2/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_1/scala2/moments/varianceMean*siamese_1/scala2/moments/SquaredDifference3siamese_1/scala2/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_1/scala2/moments/SqueezeSqueezesiamese_1/scala2/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_1/scala2/moments/Squeeze_1Squeeze!siamese_1/scala2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_1/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_1/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_1/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Hsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_1/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese_1/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
(siamese_1/scala2/AssignMovingAvg_1/decayConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_1/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_1/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Tsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_1/scala2/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Ksiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
"siamese_1/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
c
siamese_1/scala2/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala2/cond/switch_tIdentitysiamese_1/scala2/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala2/cond/switch_fIdentitysiamese_1/scala2/cond/Switch*
_output_shapes
: *
T0

W
siamese_1/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_1/scala2/cond/Switch_1Switch siamese_1/scala2/moments/Squeezesiamese_1/scala2/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala2/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_2Switch"siamese_1/scala2/moments/Squeeze_1siamese_1/scala2/cond/pred_id*5
_class+
)'loc:@siamese_1/scala2/moments/Squeeze_1*"
_output_shapes
:�:�*
T0
�
siamese_1/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_1/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_1/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/MergeMergesiamese_1/scala2/cond/Switch_3 siamese_1/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_1/scala2/cond/Merge_1Mergesiamese_1/scala2/cond/Switch_4 siamese_1/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_1/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/batchnorm/addAddsiamese_1/scala2/cond/Merge_1 siamese_1/scala2/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_1/scala2/batchnorm/RsqrtRsqrtsiamese_1/scala2/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_1/scala2/batchnorm/mulMul siamese_1/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_1/scala2/batchnorm/mul_1Mulsiamese_1/scala2/Addsiamese_1/scala2/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_1/scala2/batchnorm/mul_2Mulsiamese_1/scala2/cond/Mergesiamese_1/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_1/scala2/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_1/scala2/batchnorm/add_1Add siamese_1/scala2/batchnorm/mul_1siamese_1/scala2/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_1/scala2/ReluRelu siamese_1/scala2/batchnorm/add_1*'
_output_shapes
:�*
T0
�
siamese_1/scala2/poll/MaxPoolMaxPoolsiamese_1/scala2/Relu*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�*
T0
�
siamese_1/scala3/Conv2DConv2Dsiamese_1/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:

�
�
siamese_1/scala3/AddAddsiamese_1/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:

�*
T0
�
/siamese_1/scala3/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_1/scala3/moments/meanMeansiamese_1/scala3/Add/siamese_1/scala3/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_1/scala3/moments/StopGradientStopGradientsiamese_1/scala3/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_1/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala3/Add%siamese_1/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
�
3siamese_1/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_1/scala3/moments/varianceMean*siamese_1/scala3/moments/SquaredDifference3siamese_1/scala3/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_1/scala3/moments/SqueezeSqueezesiamese_1/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_1/scala3/moments/Squeeze_1Squeeze!siamese_1/scala3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_1/scala3/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_1/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_1/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Nsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_1/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
 siamese_1/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
(siamese_1/scala3/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_1/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_1/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Isiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_1/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese_1/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
c
siamese_1/scala3/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_1/scala3/cond/switch_tIdentitysiamese_1/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala3/cond/switch_fIdentitysiamese_1/scala3/cond/Switch*
_output_shapes
: *
T0

W
siamese_1/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_1/scala3/cond/Switch_1Switch siamese_1/scala3/moments/Squeezesiamese_1/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_1/scala3/moments/Squeeze
�
siamese_1/scala3/cond/Switch_2Switch"siamese_1/scala3/moments/Squeeze_1siamese_1/scala3/cond/pred_id*
T0*5
_class+
)'loc:@siamese_1/scala3/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_1/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
siamese_1/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_1/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese_1/scala3/cond/MergeMergesiamese_1/scala3/cond/Switch_3 siamese_1/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_1/scala3/cond/Merge_1Mergesiamese_1/scala3/cond/Switch_4 siamese_1/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_1/scala3/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese_1/scala3/batchnorm/addAddsiamese_1/scala3/cond/Merge_1 siamese_1/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_1/scala3/batchnorm/RsqrtRsqrtsiamese_1/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_1/scala3/batchnorm/mulMul siamese_1/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_1/scala3/batchnorm/mul_1Mulsiamese_1/scala3/Addsiamese_1/scala3/batchnorm/mul*'
_output_shapes
:

�*
T0
�
 siamese_1/scala3/batchnorm/mul_2Mulsiamese_1/scala3/cond/Mergesiamese_1/scala3/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_1/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_1/scala3/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_1/scala3/batchnorm/add_1Add siamese_1/scala3/batchnorm/mul_1siamese_1/scala3/batchnorm/sub*
T0*'
_output_shapes
:

�
q
siamese_1/scala3/ReluRelu siamese_1/scala3/batchnorm/add_1*
T0*'
_output_shapes
:

�
X
siamese_1/scala4/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese_1/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/splitSplit siamese_1/scala4/split/split_dimsiamese_1/scala3/Relu*
T0*:
_output_shapes(
&:

�:

�*
	num_split
Z
siamese_1/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_1/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/split_1Split"siamese_1/scala4/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_1/scala4/Conv2DConv2Dsiamese_1/scala4/splitsiamese_1/scala4/split_1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_1/scala4/Conv2D_1Conv2Dsiamese_1/scala4/split:1siamese_1/scala4/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_1/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/concatConcatV2siamese_1/scala4/Conv2Dsiamese_1/scala4/Conv2D_1siamese_1/scala4/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_1/scala4/AddAddsiamese_1/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_1/scala4/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_1/scala4/moments/meanMeansiamese_1/scala4/Add/siamese_1/scala4/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_1/scala4/moments/StopGradientStopGradientsiamese_1/scala4/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_1/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_1/scala4/Add%siamese_1/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_1/scala4/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
!siamese_1/scala4/moments/varianceMean*siamese_1/scala4/moments/SquaredDifference3siamese_1/scala4/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_1/scala4/moments/SqueezeSqueezesiamese_1/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_1/scala4/moments/Squeeze_1Squeeze!siamese_1/scala4/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_1/scala4/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_1/scala4/moments/Squeeze*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_1/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Csiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_1/scala4/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_1/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
(siamese_1/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_1/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_1/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Tsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_1/scala4/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_1/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
c
siamese_1/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala4/cond/switch_tIdentitysiamese_1/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala4/cond/switch_fIdentitysiamese_1/scala4/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala4/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_1/scala4/cond/Switch_1Switch siamese_1/scala4/moments/Squeezesiamese_1/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese_1/scala4/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_2Switch"siamese_1/scala4/moments/Squeeze_1siamese_1/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_1/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_1/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_1/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese_1/scala4/cond/MergeMergesiamese_1/scala4/cond/Switch_3 siamese_1/scala4/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_1/scala4/cond/Merge_1Mergesiamese_1/scala4/cond/Switch_4 siamese_1/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_1/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/batchnorm/addAddsiamese_1/scala4/cond/Merge_1 siamese_1/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_1/scala4/batchnorm/RsqrtRsqrtsiamese_1/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_1/scala4/batchnorm/mulMul siamese_1/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_1/scala4/batchnorm/mul_1Mulsiamese_1/scala4/Addsiamese_1/scala4/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_1/scala4/batchnorm/mul_2Mulsiamese_1/scala4/cond/Mergesiamese_1/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_1/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_1/scala4/batchnorm/add_1Add siamese_1/scala4/batchnorm/mul_1siamese_1/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_1/scala4/ReluRelu siamese_1/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_1/scala5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_1/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala5/splitSplit siamese_1/scala5/split/split_dimsiamese_1/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese_1/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_1/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala5/split_1Split"siamese_1/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_1/scala5/Conv2DConv2Dsiamese_1/scala5/splitsiamese_1/scala5/split_1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_1/scala5/Conv2D_1Conv2Dsiamese_1/scala5/split:1siamese_1/scala5/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_1/scala5/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_1/scala5/concatConcatV2siamese_1/scala5/Conv2Dsiamese_1/scala5/Conv2D_1siamese_1/scala5/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese_1/scala5/AddAddsiamese_1/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
�
siamese_2/scala1/Conv2DConv2DPlaceholder siamese/scala1/conv/weights/read*&
_output_shapes
:''`*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_2/scala1/AddAddsiamese_2/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:''`
�
/siamese_2/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala1/moments/meanMeansiamese_2/scala1/Add/siamese_2/scala1/moments/mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
%siamese_2/scala1/moments/StopGradientStopGradientsiamese_2/scala1/moments/mean*&
_output_shapes
:`*
T0
�
*siamese_2/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala1/Add%siamese_2/scala1/moments/StopGradient*&
_output_shapes
:''`*
T0
�
3siamese_2/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_2/scala1/moments/varianceMean*siamese_2/scala1/moments/SquaredDifference3siamese_2/scala1/moments/variance/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
 siamese_2/scala1/moments/SqueezeSqueezesiamese_2/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
"siamese_2/scala1/moments/Squeeze_1Squeeze!siamese_2/scala1/moments/variance*
squeeze_dims
 *
T0*
_output_shapes
:`
�
&siamese_2/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
dtype0*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_2/scala1/moments/Squeeze*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_2/scala1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
ksiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Nsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Csiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_2/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_2/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( 
�
(siamese_2/scala1/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_2/scala1/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_2/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( *
T0
�
Tsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_2/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese_2/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
c
siamese_2/scala1/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_2/scala1/cond/switch_tIdentitysiamese_2/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_2/scala1/cond/switch_fIdentitysiamese_2/scala1/cond/Switch*
_output_shapes
: *
T0

W
siamese_2/scala1/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_2/scala1/cond/Switch_1Switch siamese_2/scala1/moments/Squeezesiamese_2/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_2/scala1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_2Switch"siamese_2/scala1/moments/Squeeze_1siamese_2/scala1/cond/pred_id*
T0*5
_class+
)'loc:@siamese_2/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_2/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_2/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/MergeMergesiamese_2/scala1/cond/Switch_3 siamese_2/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese_2/scala1/cond/Merge_1Mergesiamese_2/scala1/cond/Switch_4 siamese_2/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese_2/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala1/batchnorm/addAddsiamese_2/scala1/cond/Merge_1 siamese_2/scala1/batchnorm/add/y*
_output_shapes
:`*
T0
n
 siamese_2/scala1/batchnorm/RsqrtRsqrtsiamese_2/scala1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese_2/scala1/batchnorm/mulMul siamese_2/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
�
 siamese_2/scala1/batchnorm/mul_1Mulsiamese_2/scala1/Addsiamese_2/scala1/batchnorm/mul*
T0*&
_output_shapes
:''`
�
 siamese_2/scala1/batchnorm/mul_2Mulsiamese_2/scala1/cond/Mergesiamese_2/scala1/batchnorm/mul*
_output_shapes
:`*
T0
�
siamese_2/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_2/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese_2/scala1/batchnorm/add_1Add siamese_2/scala1/batchnorm/mul_1siamese_2/scala1/batchnorm/sub*&
_output_shapes
:''`*
T0
p
siamese_2/scala1/ReluRelu siamese_2/scala1/batchnorm/add_1*&
_output_shapes
:''`*
T0
�
siamese_2/scala1/poll/MaxPoolMaxPoolsiamese_2/scala1/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*&
_output_shapes
:`
X
siamese_2/scala2/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_2/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/splitSplit siamese_2/scala2/split/split_dimsiamese_2/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:0:0*
	num_split
Z
siamese_2/scala2/Const_1Const*
_output_shapes
: *
value	B :*
dtype0
d
"siamese_2/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/split_1Split"siamese_2/scala2/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese_2/scala2/Conv2DConv2Dsiamese_2/scala2/splitsiamese_2/scala2/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_2/scala2/Conv2D_1Conv2Dsiamese_2/scala2/split:1siamese_2/scala2/split_1:1*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
^
siamese_2/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/concatConcatV2siamese_2/scala2/Conv2Dsiamese_2/scala2/Conv2D_1siamese_2/scala2/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese_2/scala2/AddAddsiamese_2/scala2/concatsiamese/scala2/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_2/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala2/moments/meanMeansiamese_2/scala2/Add/siamese_2/scala2/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_2/scala2/moments/StopGradientStopGradientsiamese_2/scala2/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_2/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala2/Add%siamese_2/scala2/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_2/scala2/moments/variance/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
!siamese_2/scala2/moments/varianceMean*siamese_2/scala2/moments/SquaredDifference3siamese_2/scala2/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_2/scala2/moments/SqueezeSqueezesiamese_2/scala2/moments/mean*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
"siamese_2/scala2/moments/Squeeze_1Squeeze!siamese_2/scala2/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_2/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_2/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_2/scala2/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
ksiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Nsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_2/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
 siamese_2/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
(siamese_2/scala2/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_2/scala2/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_2/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_2/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
_output_shapes
: *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
"siamese_2/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
c
siamese_2/scala2/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_2/scala2/cond/switch_tIdentitysiamese_2/scala2/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_2/scala2/cond/switch_fIdentitysiamese_2/scala2/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_2/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_2/scala2/cond/Switch_1Switch siamese_2/scala2/moments/Squeezesiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_2/scala2/moments/Squeeze
�
siamese_2/scala2/cond/Switch_2Switch"siamese_2/scala2/moments/Squeeze_1siamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_2/scala2/moments/Squeeze_1
�
siamese_2/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese_2/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese_2/scala2/cond/MergeMergesiamese_2/scala2/cond/Switch_3 siamese_2/scala2/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_2/scala2/cond/Merge_1Mergesiamese_2/scala2/cond/Switch_4 siamese_2/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_2/scala2/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese_2/scala2/batchnorm/addAddsiamese_2/scala2/cond/Merge_1 siamese_2/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_2/scala2/batchnorm/RsqrtRsqrtsiamese_2/scala2/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_2/scala2/batchnorm/mulMul siamese_2/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_2/scala2/batchnorm/mul_1Mulsiamese_2/scala2/Addsiamese_2/scala2/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_2/scala2/batchnorm/mul_2Mulsiamese_2/scala2/cond/Mergesiamese_2/scala2/batchnorm/mul*
_output_shapes	
:�*
T0
�
siamese_2/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_2/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_2/scala2/batchnorm/add_1Add siamese_2/scala2/batchnorm/mul_1siamese_2/scala2/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_2/scala2/ReluRelu siamese_2/scala2/batchnorm/add_1*
T0*'
_output_shapes
:�
�
siamese_2/scala2/poll/MaxPoolMaxPoolsiamese_2/scala2/Relu*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
siamese_2/scala3/Conv2DConv2Dsiamese_2/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

�
siamese_2/scala3/AddAddsiamese_2/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_2/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala3/moments/meanMeansiamese_2/scala3/Add/siamese_2/scala3/moments/mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_2/scala3/moments/StopGradientStopGradientsiamese_2/scala3/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_2/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala3/Add%siamese_2/scala3/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_2/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_2/scala3/moments/varianceMean*siamese_2/scala3/moments/SquaredDifference3siamese_2/scala3/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_2/scala3/moments/SqueezeSqueezesiamese_2/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_2/scala3/moments/Squeeze_1Squeeze!siamese_2/scala3/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_2/scala3/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_2/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_2/scala3/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
ksiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_2/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
 siamese_2/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
(siamese_2/scala3/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_2/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_2/scala3/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Tsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_2/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
"siamese_2/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
c
siamese_2/scala3/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_2/scala3/cond/switch_tIdentitysiamese_2/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_2/scala3/cond/switch_fIdentitysiamese_2/scala3/cond/Switch*
_output_shapes
: *
T0

W
siamese_2/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_2/scala3/cond/Switch_1Switch siamese_2/scala3/moments/Squeezesiamese_2/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese_2/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/Switch_2Switch"siamese_2/scala3/moments/Squeeze_1siamese_2/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*5
_class+
)'loc:@siamese_2/scala3/moments/Squeeze_1
�
siamese_2/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_2/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_2/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/MergeMergesiamese_2/scala3/cond/Switch_3 siamese_2/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_2/scala3/cond/Merge_1Mergesiamese_2/scala3/cond/Switch_4 siamese_2/scala3/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_2/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala3/batchnorm/addAddsiamese_2/scala3/cond/Merge_1 siamese_2/scala3/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_2/scala3/batchnorm/RsqrtRsqrtsiamese_2/scala3/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_2/scala3/batchnorm/mulMul siamese_2/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_2/scala3/batchnorm/mul_1Mulsiamese_2/scala3/Addsiamese_2/scala3/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_2/scala3/batchnorm/mul_2Mulsiamese_2/scala3/cond/Mergesiamese_2/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_2/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_2/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_2/scala3/batchnorm/add_1Add siamese_2/scala3/batchnorm/mul_1siamese_2/scala3/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_2/scala3/ReluRelu siamese_2/scala3/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_2/scala4/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_2/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/splitSplit siamese_2/scala4/split/split_dimsiamese_2/scala3/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese_2/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_2/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/split_1Split"siamese_2/scala4/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_2/scala4/Conv2DConv2Dsiamese_2/scala4/splitsiamese_2/scala4/split_1*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0
�
siamese_2/scala4/Conv2D_1Conv2Dsiamese_2/scala4/split:1siamese_2/scala4/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
^
siamese_2/scala4/concat/axisConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_2/scala4/concatConcatV2siamese_2/scala4/Conv2Dsiamese_2/scala4/Conv2D_1siamese_2/scala4/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese_2/scala4/AddAddsiamese_2/scala4/concatsiamese/scala4/conv/biases/read*'
_output_shapes
:�*
T0
�
/siamese_2/scala4/moments/mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_2/scala4/moments/meanMeansiamese_2/scala4/Add/siamese_2/scala4/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_2/scala4/moments/StopGradientStopGradientsiamese_2/scala4/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_2/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_2/scala4/Add%siamese_2/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_2/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_2/scala4/moments/varianceMean*siamese_2/scala4/moments/SquaredDifference3siamese_2/scala4/moments/variance/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
 siamese_2/scala4/moments/SqueezeSqueezesiamese_2/scala4/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_2/scala4/moments/Squeeze_1Squeeze!siamese_2/scala4/moments/variance*
T0*
_output_shapes	
:�*
squeeze_dims
 
�
&siamese_2/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_2/scala4/moments/Squeeze*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_2/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
Nsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_2/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_2/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
(siamese_2/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_2/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_2/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Tsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_2/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_2/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
c
siamese_2/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_2/scala4/cond/switch_tIdentitysiamese_2/scala4/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_2/scala4/cond/switch_fIdentitysiamese_2/scala4/cond/Switch*
_output_shapes
: *
T0

W
siamese_2/scala4/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_2/scala4/cond/Switch_1Switch siamese_2/scala4/moments/Squeezesiamese_2/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese_2/scala4/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_2/scala4/cond/Switch_2Switch"siamese_2/scala4/moments/Squeeze_1siamese_2/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_2/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_2/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_2/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_2/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_2/scala4/cond/pred_id*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese_2/scala4/cond/MergeMergesiamese_2/scala4/cond/Switch_3 siamese_2/scala4/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese_2/scala4/cond/Merge_1Mergesiamese_2/scala4/cond/Switch_4 siamese_2/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_2/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/batchnorm/addAddsiamese_2/scala4/cond/Merge_1 siamese_2/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_2/scala4/batchnorm/RsqrtRsqrtsiamese_2/scala4/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_2/scala4/batchnorm/mulMul siamese_2/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_2/scala4/batchnorm/mul_1Mulsiamese_2/scala4/Addsiamese_2/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_2/scala4/batchnorm/mul_2Mulsiamese_2/scala4/cond/Mergesiamese_2/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_2/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_2/scala4/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_2/scala4/batchnorm/add_1Add siamese_2/scala4/batchnorm/mul_1siamese_2/scala4/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_2/scala4/ReluRelu siamese_2/scala4/batchnorm/add_1*'
_output_shapes
:�*
T0
X
siamese_2/scala5/ConstConst*
value	B :*
dtype0*
_output_shapes
: 
b
 siamese_2/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/splitSplit siamese_2/scala5/split/split_dimsiamese_2/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese_2/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_2/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/split_1Split"siamese_2/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_2/scala5/Conv2DConv2Dsiamese_2/scala5/splitsiamese_2/scala5/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_2/scala5/Conv2D_1Conv2Dsiamese_2/scala5/split:1siamese_2/scala5/split_1:1*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
	dilations

^
siamese_2/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/concatConcatV2siamese_2/scala5/Conv2Dsiamese_2/scala5/Conv2D_1siamese_2/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_2/scala5/AddAddsiamese_2/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
�
ConstConst*��
value��B���"����<��:=	�=������X:(���Q���,�8��<�n�P,���O< m���O=R,!�#	5=�����=�K�:���d��H��<}O=Pb����߼ ވ��������c�= y�;�<�=4v!�H�z<�����Q���ֽ ����24=���=خ=�� �9 p)8�&`<�:P�R=��5� T=�z~=�#>x�<Љ�"ԅ=�MD<ۃ=(�*=@4����W
�@��<@���@-m�p,��t�=�|=۲�����p<����ܱ��\Z�<@Te=�]<p��)<=�_'�8o�=X��<�H��i=� =gy��E���Џ��}��}�=�>>=t䄼��=�_�f��=1�ƽ� �=��;
�*���;��*=Л,���d=�~�=��ǽ�h=��-=�n�=�1�p���詻< ��9���-�j=������<&�= Pl<V�+�����H?�� Y���M�=�(o� 49�O�=�!N� l=`Ͻ<�Ǳ<J�=��_��_}=�S�=*�j��e��&���؍��e=����41c=pcڻQ�����H\B�S���m��� ����Wi�Ά�=\J=mF�;1��Dk#���2�2=�f�eZ˽������=@Ɔ��:9�n���Ľ����0=�bc�����Ԋ��`wr;�q�T���Ͷ�ԿT��c�z�q�-���v/�؁ �@�뼲rE��+H;�0����-<�6=-�m/�����=�@��=b<zZ�3�4=py�<�(��0l���	���s�%�=P��<�5E=��[�@��:�:��N쀽�;��0����<�H��L滼�X<�:Q�� 6���g��� �`Y�;���b4�P�y<:��p�%<��a��	��a< ������� �.���9������<��Ҽ��[=v��ս=�w�H�;<^���=��z��?�J�;Г<,��=<c���3=Z߽�R�<�Yü�μ?�D=\��J?#�*=�D�� �C;�1���/=@Ƿ::Yv�+o,=���;�l�<~�'= 7�:��)���;zLP��3��L�= �y9p+<@u�;���;�,�<�On= ���B�<�Y�x�=Xtn����@w�:����կQ=����pW�@o���6�0Q`�?j=8�O<*n�=�3}��t�<X������1�0�H=J=���= �=$/&= ;�{�O=*-=�朽bE=�&E����<��{=�b>87=���;|*�=�KT=@�h;��=@缰���j<�}��{=ⶆ�A�����;�S=��콠2p�Щ�<p�< ���a=�Ï=�֮���=<����������=˼y=�󼊕�=:�=����N꽈z�Hgݼ:�	> d����<�~�=�M���IQ=8��A�>2�;�V�#�w~�=���=XOv<DE�=P�=�,��dϝ=�]=��|=��g���D� ���;U�P�8�
=�a��w7=_�= <RrS� ��|�@;�5ɬ=���ԕ�<΍�=Z/�⬕=( ��	�;�e|= 5���<=8X�=N;�2���?�C=�~C���=X�W�X�=x��<�����9;P���������[��G��1��G�>�ۖ=�'x���ŽN����P�5jN=J�oѸ��6�Tt!=g��0��*��~ؽX|��&=��|��89� 1:d��<��<\���������Ŋ�f�<�\�� 柺��ּX�μL��`��<�]�H��<�E�=��<9��3ܰ=rz)��p�<>�-�0��<�׊<�����<hF��'V��U=�>n= =\h�X�I<l�%���`�<��ļt=�,�<�-��@<�����~;���Je���� u��n�m�03�=,᰼�{�<�_6���; ]=��5;�/�<�����4��$�{�'=���q�l=�ʛ��#�=pd��X�~<��H�8��=t���xX<&P=	�_=dĈ� ŧ<����z�< ��(�4��ݫ=@��:"��5�U=Hlb��w=��ʼ��=X�'<3������<x��<��<�@�<�ቼ����Ko��63�$jƽ��u=��7��N=��R<�.M��s=0��=�T�<t��<����T�<�=���ܽ�=�)��sP<����0�λ<�m����<�O���E�= G�;ZJ�=<~%���u; -�f=����l�@��{=j�=�u�=�)�=8$���	�=`XX=�+d�� =>-����;i16=&�d>�-=`�;�#�=H�Z=�C�j�P=d�?����� �Ựי��6�=����B����|����:��p���<0=`M�<���<}wi=f��=�3e<h>P<��1��u��s��=��=X3ͼ"��=��=���{z��4I�����>�*���o=���=8K�D�=Yy��MB>.�������=}��=��<�.�=U?�=��V�L͂=ϸ3=�+D=�0<XAj�	N�@�&� �j� q2<"�m��O�<�=p~��Q�P{�;lz�8躼��=�;��d�<���=dա�6$�=\Z+���(��D8=x��<ܴ�<��`=������;���=~L�r�<<T��r	�==�>N����<��9<����]��@�¼ Q;e�=<(�=,1��q�޽|�C��Ɇ�=�p߼�W����;�4=�R����E��Ay�{>���E�,��<(r� ��� �<p�<
�=dܼ朼���<V���@�;ᆋ�@������ �h���J;=��;��=6�=Ƅ=Z,D��e�= ��`~�<�U��G�Ȍ���r�$�|=��h�w<A�<�:~=�@ �0l0<��=��q�D�L�Dp�<T֔<�P=ș�<���<�X<x᪼@��:<a�<����P����5��:�y�>��$��B=��&���<��=�f�<>�=$�$���E�����n�<���<��3=L×�6��=�	�< ��<�_�,��= z~;}ݼ�Z��$�=��v<�Ķ�df�<&����<0�<��V�1��=�_/=�N��=(�H�&w@="Lb����<�y�<&�v� =�9`��<�JA<8[	<�O���w�Hݒ��A�6+��2sg=�C_�P�z=�!����d�;��=�U=�ZL=8X��@����},�sI��E�2=�4�;ؠ�����@�U��������<|s��==�
<YWZ=��7� D9��|Tļ��ս��a��Щ<�b�=���=��m=8,x�Ɛ�=-8=���� =-���e��X�<PC<>�=h_޼�F�=�ٌ=`<K��@�<�#���4/�l(5��E��W��=`/�g���5�~yG�$D��J#���<P���佘<F�==�=��=���\as��Z��@�3=ݻ$=mx��=4_=�;Q���<�Ёq<��=�=�� �=���=�����=�dܽ尵=�q����B�=6��=��<�ݣ=�hr=�[����<l�<�d�<�q=N�������d+;�2ҽ�B;����}���=�s6;2��x���*��
 � �;:X�ռ q�:Z@{=~e���=��L�pP�; 	`=*�<Ȋ;<���<��9:�i�<I�U=��� 2�<����<�ή<����<�;<>-.������%����<��= �"<8�3=������Ǻ s�;��<�}�;�㯼 �<J�<��ۼ��L� ���|���!缼�< ^M<�o�<��< ��<^�=���@���F M=�.��<x��0��<�H��}�P�;S?0=��ݻM�<�$s=H/<j���r�=D�= ����D]�@�3��-񼴵Ǽ�~�=Px<!ZA=�� =F�=�;ּ��=��)=,�����`�hoc<x��<!�P=p���X�)=ؗ�<�"Ļ 4���=h�W<h&U<��>�0?<�DP�=Hu<��P=l꼸k�<�D=��<2r(=���(
���)/:0'�;X&=��=Pڅ<�c�=5 =���<0V�;4=t��<p��	���7=8��0�Q,=p�<���<���<�����);=ʍh=��;�̣=8@�<@:=*�I�X��<hV�<@����s���3<x��<P�v<0'Ǽ0���+���������Xk<�S��哅=��%���&� �߹}h�=`�u<!.~=�)�`"5�8Eq�v*e�_6=X��<���%��>������`�; �(��6i<�|%<4=`����� �ϻ��e�)�ν�N� $��Z�=�pr=&"= ��K`=�m�<ʼ���<�X<���@#�<:e�==��V��N<=��>= En����x��%��Θ�������=������x9�t��~4��r9�05V<�=�������< .=�#=�;�H.�� �e�@B���;�v���վ=0��<|��<[8��)����=��=�.[�6=e�T=N��|u�<U���@=t���6�#�R�=a��=`G>;�==�^o<�!_�@Ϫ:� �@���Q�=>I��vC��^����ѽ�� <�U���|?��2=p|�<P����8�dL��B�)������Y�@1Y�a�>=&d��h<��@��We<��r=j=��x�Pt� �"<���<�b=��ȼP<�tżD0ڼ�X�;���;���<8�a<��< ?�9�<�y�<d��<��"�)}�=��O;�%�; 7�:�Ƀ;�z<�e=@y�: f��n�<����96<dE�<�1��hA �ޢ= �<�n:;%YR=�B�: $1<�/<b�)=D��<�<�<=�<���<a���@�P�;�c�<���)<_:\=x֠��C
�d6�<�OJ=0��;8X���|�$J�� L8���=�/=�s�<��<�~@<pp��D��< �O;�L�;l_�<<�&��0<�d6=(R
�}K<tO)=L5�< 49�ѻ I&<��;���:�[�<���<u�<nv=	���9ٻ$�ܼ�ֱ:�=(c��,�< u�;H�1< z�9�= ;�:Ϫ>=4L	=��<�"N=@�E�0 �< :�:<�p�M<�sK�H�9�$�=�^�=�AH<��<��)�8�l<��== �k<�O=�{�;��<HM���_�; ��:3�!=8и� �K���Z:�@�<�a���Ƙ��k��E��`�A��]
�<sE�Y8�="�#�[<Ħ�<�M= $���y5=.:0��?�� m���ݼ.=�v<������� ȼ��_��e�;�()� @.� �;s�=��R�(�,���<ؖB�c�����%�,?��j�<}�x=��$=�4��D8a=���<8�|�P�$=���< �й� ==��=�R�<�^��;= W!<�]��� �l���푼�]�������&	=4KǼ~�ӽ �g�A���{d���pG�<�?�8�c�P?<@��<�%=�r������ �̻��P��P�<'�=���;L��<����R���r=Zh�=(I���=�~0=���� �_;b��l�;0����b;�`��<W�~=X>t���<�<���< Z��0t���Y����<�&����pԊ��*{��d�<�&.�D1����;��<�깻(�>�@������l�����ȼ7<��=���0��<���(�<�}=��(=�$���`�0r�<�N#=H<���`W�<0^ļT~��s_; bA;݃<��*<�y=����췓<0_�;@�w�z�U��Í=8<(<h�n����$�;Zآ= �?�  ��d�
=�k�@K\<���<�����7���=����ן�~@= �q�L��<��H<��=XU�<LƋ<���<���<�̼�~�ʊ��-T<�W�� 	(�.�6=8������	<%�*=���<X&��<������:�\��<a�J=��6<tX�<�LԺ@ڽ���<d=����<P��<x%p�p��;�<9=x��� ���r�	=�}5=8 < �ܼ�t�; �841�<2ZG=�=��@�_<,q�<]#��I���#���3;�<HI�D�=`:[<@i��� n��H=Xpܼ$��<Lg�<�m;|ۆ=�H����<0-»>�=���<Lkq�p���*�=�_�=`%4���<Ĉ��hU��|�<[ <(�<h�T����< ���X�N�Կ��(�l=�׹��
Ҽ��2=�p=P��<����`m� :�����</ܼ=@=܀��Q��<�;`bT�4Ϟ<��� K�<�޻��$=��<�5������=���=�'"��
�;\�ټ�Ə���;�A�< Q��9��=xˎ��=��������׃��=,�F=6-�=���=�����=��� ��;n$m��.-<��A� .g<�}=���="0=�|�< �=Y�=�ʓ=5=��;8!���4=��u=������:�P�;U}>=V)b=��`��s�C�<�<Pmü$9�<��/=@�V��M����=L�üī�=��<�� �z���3=�ᇽ��[�_=�q���A$=���<�#�|�.=lӋ�옂=��̽���=���;v� .;;�<��L���=>=�=Ҧ��"*=�<L=0�=,�� �6<&Wi= �\�@�;e�I=��c�=9`�=�ƛ;�]e�X/�X�<H�e<{�=py���<���=����c�<8%�<zy=.�+==��^��=��=>�E�Rt��"r��1�ýZ�/=)^�[[!=���ֽ+��ʹ��D�M��X���诹��O�=���<,o��� �ۡ�V<=&�z=
pd��y#� *�:�^�<�2�� �<Uܽ-X�� �u����=�r�0����1��`P%�C׼��۽&�(�@�9�j=��R�9��� ��X&߼�.(��,0�0�R<ߣ��@�(�ȕ�<�߻��ｎ�>�������߼�ѵ.=pa�;�P��~<��<)� )=�=��<hp=n�w��T8=�F�<䁙�C�=tԊ�|�<�ý�v�;`�����( <���< �4�X��<��˽L!����x=ҥ#�8y1�8K���P��\� =�VP�ȴ�@�;jIr��p�<@񄺐�.<��=��ý�H==��Ƚ�>�<�/k��4�>����U���i<0�6T�=H�[��:<n>潂�H=��<���'��M$=�bZ�v�����=Xr��@�ɺ0����p�= u�:	���T�=�U�<�y�=vg=�>��@r,�0IY<�<K��(��� �<�L�<lۼp��;�|��U�<0(�<��<�L�<��n�=  �5�{$��k�:���<Kg�=��d�$φ<b������p☻Bb\=����0�=`/;��]=fY����l�����GC<�~=X�>�S�=@=n�=h5�< 8�<+�����N<
a5� 8�	L=У)>�(= %"=�R=�Ǡ=���<4��<@_��$����ϋ=�=kw=x_���ü���<k�&=���#=@̚<W`9=T7����`=�ҁ=���X&��r=�k`���=��t= �n�j\<=4r�=�*l�]�˽��λz�p�x��= �r� س����=맽2P=�����>�⼶��I;k=��=���<��=�4�=
���΄=kLp=k�= 0���Z��GH=����z��=��2��=��=`j<$Ř��'�<�,!<���;2/�=��/��`�<%�=����0�=%���&�<�3E=���ځ�=��=��z���iۼ�x����=>�5��]K=��<g���6dy�\��/����J�l�t�pJ�;��>��=zY3�Xx�|K:���o=z��=�P��i'��3�;m=n>ƽX=	�˽c ��NX��8 �=D'/���r�Pc��@Z����Uʽ���l����
���c&�X�t<T����:�اg�FW&=�/m�$�<�P=`��<k��aL>���x���)7ͽ�<@f�;�������I�Y9=�i4=c�Q=�=���\�c=���<*락��F=\0m��
�<��S����<h�"�vh��`<==��K����;b?Ž��Q�/��=�� ���/����� �=��u<��;�Z�B6k����;�e�<0�~<�K�=R���L�=:��<'���h��<�:7��[��l��<��;:��=�8�(C�����~/=�ܼX����4�=(������X��=�[��0	�<�/>�r�=��5<p�Y�y=�s=H#�=�=h(��8&X�0��;z�H�@�ٽ�W= P�9��J;��5<������ =d�M=K=�=�(��p�V<�'������ =�z�;�\=`j�ȠD<�Q��x�P<.��nL�= =��em=ȗ&��UG=�hh�@��?ۻ���M�qU�=ֹ>��=���=0��<1�i=�%=ZL���<>�� ���2qG=Q�=>��3=�c6=-�\=L�=PY��=$�����޼1�U=w�<���=,#Y��KW��z"<`��;&8O��;�<$\�<l@=��;���=�R�=Ȑ����� �D;�qw�J��=؁�=��¼��=I��=�O1<�u���x-�����=�ﾼ�W�<n��=&M����A=\.���>~���}�G$�=f$�=���<��=j�=�N5��*�=J�=���=l��<��q����<�=׻L�W��J�<j�\�h0�=�j�= �i;hU��&-=wD�؂W��}�=���Xվ<���= @/�=��=��3����<��U=���;,k�=Ee�=��5�h����<~[d���<�p6���b=�U='摽������­��L���c��&�<Z�>�Q�=Ȣ����
��$��{H=���=����d���<]�/=ǽ��<�������� i�6(�=�̼X�}���V<@m�� �^<���D�ݼ0�;�� �:���P<�U���hA��4<8�O=̆���=$ɋ=��O=����}�=Z��_������Ӽ��ټ����P�<���)Q=�=8��=@ �;��;�Cp=�^5;�Ry��YR=�� �&=P����]= ���j���=��O�h=X�r��FB;�$ʽVH�~<>0ܕ���?<P��;��; ��=�4=$�<�֩�-x����s;���<��=n��=,���}�=x���p�<�f���G=�Qx�(K����<�.=�t=Z7�Fּβ���<�D��Hpj�d�=���; ǐ��;�=0�<�;=f)
��-;=�tU<)���`=�	=�|�=le�<|k��̚�����;�l-�|#���xw=�������<0��<k޼��<�f�=��K=GĊ=�aM�  (���������1@=x�5<�%T���y��r0< ����F<Pw��{�= �;�h=��N�v�=H�)���M��h½��t�<dQ>1[t=^d=�F�<N{f=�U�<n�L��l�<��Ż`�K�N=%c5>Q�`=�5�; ��=��=���@�P;P�D��{3� �:.�.���=o��彄A�<��2�����@Ϟ:@+�:P1�;��2:`�t=޽�= ��;95�(%��Ҕ��<]=ŒD=�꼦��=f�=b�<Y��x��OL����=�F�� N=��=���!<S=���m�="��������=��=��<d2�=A�=H ޼�c!=0�<�+=��T=�x��HK��(�<���D<�ϥ� J�<w��=8|`<�]��r�;`���������=��Q����DpM=Tl�C�-=�р���<�9�=�y<mj=X�m=��;��e�`z}<ƭM� 'e<�8\�@�X<�g;<�P��\�H���G��d�h���!.=�S�=p$�<>\
=W����n��5=��/=�4(��A����<�=��p�hTF<�>.�����|U,��U=0�V<���<��<(�+�t��<��'������G=l�ý ��<�M��=��k��X�ȟu<�9=��м���<3�a=��<�酽)#�=�,�;oX��m����<�h"5�"�
�$t�<�l�ͩ�=u�)=�g3=`Tʼ��<(�h=���<�μLW�<��B;O�4={	�4�=���%�<K���i=(�9<�,�;�t�� z��w� >0X<���<�P<�.�;j@�=�h2=��
=�ݫ��H'�h�y< ^�<�B@=��=�I�<H2�=`|h;�r�<��ͼ�՗;H�u<�x˼�N< 6_<;���f��������I< ��9<a��Y�`=��=@"P�@��=�9=gM=����0=��; �s����;���;V��=,�<:=���ּ�\B��~�L�b���<��� D*=F�)��}�@�6�{��=z�=4��=�.��Wi��B�i���_�)=6�<�E���#�X�����@�;����=�B�<��W=����0<�nż � ;��Ƚps���rc����=pG=�w2=�U�<v�D= rO:� ����<��s<����Pn�<�D>Fnm=@��[HN=$�=�>D�ȋ��Kb�IG�p7�-���(��=�ɋ����(�,<�V���6�0A���n?��D�d���&;=�\y=���<l�L��]�dk�����;Phz<<`ϼ���=�"?=hU=��e�p؅�ȜO<�&�=�SO�s4@=T�="�� =��S�F= 䜼�����=���=Ї�<Ƥ=�{!=���:��1<�k�;�;|<��}=�[½b*�� ��:L����O<5������ʙ=Y	=@��;���o#<�=�ܧ�<Ɓd�_[���=�`5�d��<�1��0?�<vڍ=Ļ�< ��:�C��xE<���;��<�K���;���,��(���p^� ��9@�� p-<V�� R���5�<�J=&�:�3Va=�����U;�X�<��w<�D!;�?�< ����j�;�Ԉ;`��1�h�1<D#�� Kd;�|=<��<05ջ�&�<��G�����+��	�=�MҼ���;�)���)=8;0� ���� ;�x�<�-ּ��<�(2=`�H��|����=�0=�=::.,�PӸ�L����-��x<49�<�Z=�Ҏ<�>,<h����S�<hP�<xo�<�ۊ<XW���� !=��޼�<ԋ�<(>-<(��,�;$��< �+�����=0<�G=�s�<&�=����R�� ���M�;j�=@�k��6<< ag��K<�;��=�?<л:=��<��<���<�����<�u̻�Qļ��9�l�O����X4�<K�= �2�t�<�rܼ�ǽ<uN*=� <슣=��<˽=X�M�(�u<|/����3=���=��7�/=$��<�)���μ�'ڼ^�ʽ���P�������=r�I��߻�';dkl=H�<�rC=�����
������vY�^�=P��<Z{��VS�(/w��׬�x�<�V��Hm<��7<��$=0ic�l�� &J���k<𭽐iѼ�����&k=��t=U=��~<dCK=��L<��Ҽ�"=�w�<@�!�X
=2w�=Ep=� R�:"=��b=�~D��@�H��0޼�@l�넄��y=T[���� U����������@�:<�;x��$��<Y�+=L�=t'���q���3�hu��w�;P���=���<T{=,
Ｐ����#<=�Z�=��f�.�9=Vlb=�R��H��<�������<xe4�,���O=��= �;2`A=@��;��{< ��9��'�(kq<T;=΋���̽�{b�����m�<�(�� �\��1�<���<Pu<$[��@��������У<!�g=܏��@��<�iY�(p<�&�=
Y=h��0ID�X�]<8W�<0T�<*��؏V<L8¼�$`��=@� @����<P뜻��<������!<�4-<��;h��Ua=�W,� ,���aʻ���:@����=��Y���D���<����;\.�<�fj��@����#=0o<�X3��j=�c��0��; �4;g�=�|<�s;���<PC=�}���B"��z� y�;�	ͼ��:�V= �¼�ꀻ�i<�]=��k< %q��QQ�� ���0����p<�D=`�; �����޻�˰��c�: �[���;��<h�|�`�;��1=���(>���L=�='= �G;�)�<K�<��?��L4<C[=���(Le<�=Pg��
�����ǉ��f�<H랼�g=`�4� o);(恼U<|=�Z��<�l�<<>�<�J=�Ҽ�C-<P�;c3�`���P���@���"�4=*��=�Oջ�`�<z$%�`w�;-5=�Ҡ;�\:= ����9�< ���3������ŋ=�����ͼ�f=z;s=@���k����!��%������${�o�=\���`�;������ɻ0��;����xs��l<�`=�D=�J��:��<\
=��=�Pϼ�=��Ѽ������h< �<��3��Ff=�1L<}P=ڨ��즂<d���ق=O�=fd�=j��=�?�@^e=��L�����z)s��(�� �� #<���;��@=k�p=�K= �~;B=	=
t�=r(=p �;�����=�1�=�����<�8�<�� =3�=lK��h��<���<�=��@;�2><P��<��)�`V�;�W�=�h;v�A=���<��'<����M,=[���JO.��Ҡ=?U���ǰ;H�J<"�r��<�+���,%=������=��;�a��@�%�x�X��1ϻ��=���=~׋���D;mre=q��=��1���<��=�c7��>=�k=h�i�J �=�l�=�'ѻ {�:�K�;$�<��<���=������;H�=0�<�r����=�c)=���;XA ���=f��=�e�n�$��˽6�R=�0��̒�< �<K}��,M�4���O������h@N<>��=��
���n��@�U��==���=v�e��4�p���2�<4�M���<��޽�l�求����=8�ʼϠ��G���m�:��������a���߼Z�нJ�����[�<`kQ�f2��pZ�Hu'<&/~����p�Pk���ؽxx<>�T+��C������B= }�;�N���^��D!��e=^�=��r;��<+Z��@�=�Yb=�뗽�$=2Ԑ����<��� P�:~⑽���(=�<�2)=8�I<^�Ƚ`�J�c�=<�ƼPꀼ ������u#=�����r[;�A�<�u��U&=dpͼD�< ɶ=������<�������<� f�����E�����<�e<<d_c��;z=�}��p\��4ə����= �d��z��1�<�M�z�C�p>�=xO�<�,�:�x<��l=v��MOl=��|= 'һ�v�=.�l=�V��H�,�x�;<�+*��D��@ŧ��;=��*� WP<�ϸ�D��<pE��\��< �t�(ږ<�-=%�<@p"�<�Ӽpp�<�`�=����6D=��������Ģ<���<���r=pۍ�[�=f�����<ʐ.��F==v=&��=�ّ=�[�<�Ո=�2��5<�6��X����^��a�DG�<��=�(n=�S�=��I<�.�=��7=Dݝ<D�<��Լ���=$�=|�ͼ�ռ�r�<��<a��=4��<��R=`7<l��=��u:{�=�=����`�s;g`g= Oʼk�=.&=���; ��݀�=L��I㰽I2p=m����k�<@��J��,��=Z%���R�<O�����=艄�.��Xp�<bJM=p�<�C�=���=�^m��=k �=���= !�;���,z�=|���D��<��=h�¼���=2�=��; *�9�#1=<��<�i�<t��=*d1��=��1=B�=!�'=� ���,=�=v<�٫�:c>�'�=tx�������+���½Ҙ>=���F=���<���p �|�4���h7�ބ�rF.=7(�=FG=��P(�8tͼ���=�Z�=��k��<.� 8�;��<(픽�l=��^�9����Y�=X�޼ZD������;�\U���۲�BQ�<�����:&��2��=';��!�\�ǼP��<�wM��KF�P��;0��;-ܽ�E@>
�B"�,/#���2=8�<Yܼ�|�ԶP��1s=Y�A=��=VZ=b�[�:Ӽ=,t~=d8���˅=2���[=o����<����r��XU=�G=�x+=�z�;�׽�U�����=��4��1��L�<Ǻ�����=�<�0< ?�<��r���<��ͺtG�<p�=��H���O=L���q=KQ��p��L�_�@�;6�<<I�<��=�jp�.�A�N�ֽ��= 9P��=F���=�����i���=��O=�� =\�<��v=�y� ��<5Qu=  :�% >aID=�"ȼ�ż@Í<� H���ս@��<(Y2<6��8�N<h5���-�<��*<�9=XE�<<�<��<P��;6o&�`@)<�c�<�{�=�����B=�彠�;($�<��K=����hj?=p��d��=������=��x���<C�=�K	>�؃=�DP=c�g=��H<��<iה����;6��P����@=���=���=�o�=2=�� >4��<�G/<��<��'�X��=�J�=x�=̣,� �8;���<�H= w<=�J=`)���X�=���;�^=��t=�����Ȟ��= �4����=�IR= B��D&=�T�= �<햽�p[�<�g�F�=�9����H�Q�=>�˽p��<���Ю
>4���ܻμ�r=�Ӫ=�=��>��=�t'��==m��=�O�=��=���|��=Ho��o�Y�=�50�y��=Ew�= �;�;W<�W�= �<@L�;��>
R���<�?R=D�<��u=dy,�g�8=D�)=\��t�=+?�=0�(��Y����ؼi��� �=_���9=��#=��x���ؽtL�vH��h���4��e�=r��=y{y=���r.�X�9��=�y�=�
R�� ��V�<���<bA���� =����I�nb���-�=�b:�<!�@�
��f�pU��=��"�#�8���{�����m�B�+=p�b��_1�@��;��*=���`i;�6�<�m=����}>�м�D���f�@)<��Ի������x~<��6�=��?=��c=�.�<�Ø�L�=%�L=^N����=H$b���E=b���==�ǽ��~��K�<=�K=�4�<г����ֽ���g�> �%�L0��P*=2*6�85�=}�-=(IC<��<"}s�8K�<��_<p�=��=���j~�=H�25=U�� ϩ�C��P���0�< ���r0l=�F�w⋽�q׽MT=X_��'�,��=ll����5����=���=�f=���;�,C=��l� -)���2= K{:�>p��<th��D����<��I�"ݽe�C=�x��y���oJ�
~�`��;EP>=|!=-9r=$"�<(�f��k��%���=0ث<dD�<pm9�%?(=� �a<�#=o�^=�嗢r�R=�7�(��=dEq�D�=烚� Z:L�<th >�@5=/�j=�BR=DQ�<@�w;�����<����`�n=�e>�	�=�)c=��b=.�,>`��;�9��P:ﻨ@H���=@�s����=�������dU(=����9'���<����<��M�K�v=tI�=6�W�d]� ^��́�~iQ=�=HzD��G�="�=0޺<�Q��h_�����w=8�i�m�<���=�� �2=>3�>��=�A��f.�G9�=i@�=5�=�[>��=tؼ�=��l=�`=�ʆ=pҝ��l�=�����ڽ�i�<J���Lq=^�=DX�<�g=v�Y=�=|düd�=3���@�:L6= Ϛ���0= K��|);=�4�=@ٲ����=/�=�c�;8`5�8��� ��`��<[���8E���<�jI��'��Pk�����ͽ��z"�=�Ҹ=���<��";���$�y�x=P'�=���E˽�L�<���<�6r� ��<����H��d@��f�=�$�<�x�:�RZ�സ���d�ƕ9����e_<�
�dE�6���f=( K�UT���<hV=��P��;���<<��<�z�P��=X�
< 3�W&�0O�����~Lp�@�];�����=�8=�7=��G�0��;�C�=��^=�&���'=x0꼨�,=�!��xf=&i����9� ��R�d=�=�[�.[ýa����=p`y< �S�]�#=X�ټY�=��O=�7�<�yO<Q&�`�}<���<�E=��=���<��=�g3��]A=��O�@�ļ�!�;�ļp��<H	Ǽ8�[<؍R��;r�خ+��l�<��Լ�_Ӽ�f�=�iB;�l߼э�=���=h�=P<��3=�r�� (Q<|˘<�ۃ�
_	>t �<nx�`e-� �9�1�\����=����uX;>Q
�^-�xԏ����=��=���= �7�LiƼ��`��P02=�٭<@�i��,��G�<��#�_�<� =��5=��Z;*D&=���b�y=���\H�<)���I�D���?�> =�<_=�[=�>�<x��~�>���<(p<0<׻��<W� >�l�=I^<�0[=6>h������x���LYI� �C<��9���=�������<^�o�ᰬ����;X�Ҽ�o<tż�AZ==>y=�����������'��8�,<��f<�]߼� �=�D�=6==!���@[���7��V=\0*��9�<٭�=j��=&(2�$L�=��ą��=�'�=�y=4'�=xM= �Y��<4�	=�c�<��=�Oֽ�=�� ��ʚ����<͍����<U��=�,=^�,=`O=c�1=�*U���=M}½��:�|�<~M���<"E��@-=��=��L�X�B<��<@!<�C��I<v�$� ��;b�/��'n���=� Ο�Z��������W<�惽`Hi���r=�oR=6O��Ƣ<�tn�������%=��=��=��O�@Р��a�p�K� C�:�_� �9��(Ķ<�=p�><x�����[<P�w���ܼt����<��T���:��>�=���<Jʼ4��<��~<�$� �8;�}�<�,;4��
==��3=p���w�����'�f�P�;�����(=8}X<��H<<=��Wg<�A-=�=`l�;��j�Ł�$y=^�'��J�<8�� <�8��3��<	�=0b���l#� �o��p�=�{�<`>�<�d�<�׼XM
=;�<t(=�)�;��<H�����<��<�7�=6�=;=�$o<̻Y=����'#���g<���`û|���B�8��.���T-=�|<����T��r�&=�, = ��6ĺ=�3a=�K=�� <�`�<3漯�V=P(�&y�pV�=0�<&8�ĩ�p׼H@	���s��/�<����m#=FmI�������qx=��=t0!=����t`���v�3����=8dW<ܺ"� t� ;^���=�� <0z�< �9�p�<�|/��t�<(x��va=������Z���ƼB�=�eE=�X�=�-=�<�����]�ھ<P��<(+����<���=�_�=s����0=���=�:6�p���F�|�����*�=�=��s�C���:��鈽g:Խ��� ���Թ�<�iڼ�7='�r=��C<ll���8���d��t� ^�;04�����=�z4=Z�"=J> �)�|n�<'(W=��l��2=���=i�`	�<b8��J=@�߻����8�=f�=�K�<���=�*�<��n<(��$�<X#4<��~=�½hTK��ѼD�Խ��<[���\�< 'O=,��< =��=���<Z�V���<mT���;�;���<�D��@	�<�ן�>`<5i=�n�;�]-�H\�Pk�; 긺Hb�<3
�@L�:����8j���܅� �*����<�g��H�<���@4s;T��<B�<u:���f�<��Ƽ��� L< )z; pg���I=t$��4��H�.<��ؼ�{#���\<�E}��Լ�_$=�<�n��{A=��d:вڻ ��L��<D���ޫ��W�<ą0=@������ �8:�X�;���@f�:40�<`pٻ@�ٻ�0O<Vf=�(ໄ׼,;���A񻶗�Pr'<�yQ<`+e;ڢ�0V��&��x�<��< J�T��<0�=�@
;D! =\���+м�t_<�}=���lk��E�<0.�@������; {�<�EW<�{=�Vu��Iۼ`�ٻ��y�j�=���j�$=��Ӽ�"�<�	A�:Y�=���<�<I<VE=�=ǳ<xk켨_:<�i<;�ܽ��x\w�����ҥ�i�= .;�]^<z,#�E�<Oy7=`9p;:3=���<ȅ�<�d��й�8�p�=�^�ռ�=I=�^="3!�����j�����%(�~����I�=� �0A�;��[�p�2���A�`�;&�!���<��=���=��ѽE�����<1��=��ۼ�f=���g���x�<�h��g,���=|�&=�B=�ǜ�p4�<p׉;NŔ=��<�x<��=�L��x=^X��l"��6�e���Ǽ{���(�o<o�h9<�c�=��<���� �;��T=�%<=@=�:|>׼�y�=�> 4Ƚ�<0�< ��;{	�=,�����=L,�<D��<�4�<��k��g�;��p�-={N=1�<�[<��=;T��<D3�����<g-��,g�4��=����P-y� `�L��`xټ��o�p��<Ȝ�m�=`�I������y�V�8� Ml��NW=&+�=2}���-�GO=TVI= ��:|l)=]�g=0���&��=pQ�<x�
�&Ŗ==Fg= ����r<��G;�<m�>=&��=Ģ��������<ѿG=R ��!=X|�<@߯���1��w=`#5<j����D�z�}񉽠Ej=�M(��x<��!<@7̼wJ���a��9�;	�����j�`�-<��=>�0�x�߼	vӽ�Ec�(�}��V�=��l���\Y��-�;P��<��^�g�8�V<�w?����<���쾄��q���_�<��*���	�V�U���6:L�o�6zD�
^*���<hB<�p�;8��<︼�
��}S�ة0�N��,����=P��;�uZ�x:���V=��#=y�½��g��k���<���<�����6<��R��=��<~��0����D*���<:$ ��������%19=�t�S|=L,� �,��P9=��<�f� aX;�<�,���]�<�+O�<Ƞ�<2-�����X�0��"��&k=`�5�Ʊ��Wü<��<L�Ｐ0���x���H!=D[���� �( �<XQ���U�;L"ͼ�a�=�
ܼ'q��P5q��?ֻeh�����<PΧ<�x"<h�{<���;L�c�1;�=���<`��Ox�=l�@=��*�TD��<Th	�<7x�\�e���`=��Ե�<�.�; ��:ؼ���n<�:���<��V=�A=b������8�q<���=�KżP��=$l��
D���<� ���r�Lr5=��;\��=�]���z7=�8�߀�=�&=�0K=�ă=����=��w���I�/ݢ����WX�����:�	����5=b��=��=����h�=��f=ܰ�<؍�<�<��g>���=&r��X�wU=ۡ<~��=�V�<�?=��;
[W=`��<PSL<`f�;m�ѽ�0=�d_= >���J1=�\<��<,f1���x=�����喽'��=q����m�����;�뀽��W<���� ��<+Sͽ�9�=H=��������<@r����<�D�=`�=��� WɻKA�=hj=��;`�i<���=@����f=�=ذ�i��=�݊=<���d�:�k=�2�< �m=�ӹ=�}	����<��_<փ�=��< yK;!�=��[���J��k=��!=F�)�,�T������1f=��E�d,�<L�<�_���$཰o3��!7�_䶽�*[�G=R�o=��`��x�Zo��MӼh�B�Ѩ�=VA�.n�����`qk; *T9�¼�3s�`�;��~��2T=�>0�����Ŀ꼘�U<.:�Z�#�J.Y�@6Ѻ�
���ts�s���<+=�.C<h���_}�0�O�teۼzG��:?� bּ"��Gq>@+�;���v�"�=�V=�2=}�۽��; L3�:(<(�<lV��$/�<I���毵=�z�<Ԫ����< 4j��� =!���ހ��w׽��O��5W=7
�:!�=J%���[���=�S=�p�h�L8���0��Ƌ�=6�!�`�#< �<z�3��������j���_=P��hOR�`缘#�<G��@j����y6=�7λ�	�A�.=�&���ļ|�o�vј=Ƚ������`�;h$������C.=Ud&=���<g�<`��;4G~�~7�=`<>{��>tX"=�g'�(l��Z=:�9�,�ͽ�;T���<��+����<�{�H_<L����:<fI����<0ot<P��<��)���:��Yf<-�=D�ּ�®=���J���89= ��9��P��9K=�C��#M�=�>��f�=4
�(0P=r�=��=��==�6�<D�=<�G��	��v�� p�ޮB�xl�`�b;5�=���=䍬=�bM<�6>tu�=�w�;�
=��=��B,>�ߦ=�<�d�g�7F=�(=���=4k�<]6=x�Ҽ�	d=�e�<�C�<p��<�W��P�*<��6=�i�CSQ=���<j�=@��;ֆ�=Vf9���ν���=b�`��ɼxu�<�j/�(�j=���$��<���|�=��XŻ��L=��"=c�.=��=�^�=\�X�,��<���=1h�=��<�)M�d�>(m	�@�;wEL=�bۼ���=�1�=PA�����<���=��<�eD=���=nb��(�<���<��q=Lu=��&�(�:=|T�<p d�T��=^qq=��ܼ�6h�P�����;L=����u�<�=���� �p,c���ʼ�r���fC�g��=�-w=@$�;��]��g,�*㼀?[�=��=^Ek�#����� �n;�ۦ��J�Zx����:�A��闂=���x�y�2Q� �:f4 ��f�Z�n����:V���[�������E=�\�;l7ɼ�$� ��;�����O�te(���
��F�^T>ܗ�< �A9��6���=L��<wrսhq�<���ȼ�<Ѹ@=�P��,�;Voi����=.�=
Cɽs�%=�{I��XB=іݽr1�
�콀8u�ӿ=���q�=��=��U�|��<��=`j1;�h�� ����hɽ���=J�@�<�f�<�=��@��i���^<cÆ= �: ;<<�!���7=���Lн� �H�P�<�Z)�����1=Pi��rZK��4i���=��ۼ�T�D�<��ƻ����2�=4��=��R=M�<��B<�=t��Ӻ=@�;-����5>�$$=������o9%=��P�������<��C�~��� <��ļ Lu:`�H<�@4< f;�O�<\Ѯ��O��4����4�<2�q=�Q�FY�=cu/��t=<��f=�4<�1�fs1=p�1�܎>����4��="{V����<���;p�>h=�vi=R�=�����1��폽0�a<ps4�>�����<�F�=��>��=B^=�W>"�G=$'���<0�1�m�>�D�<�4=_^�����N8s=��m= �\�P��<26N��j@=�G<Z�=�;=���,��u<�`i��=x�<�ɾ<�{=���=̈���0ѽ�]�<��
��ʧ�$��<0eػ�z�=�3"�؅�<R�;�a�=@;���"1���=?�=DP=��>���=p����T�<�-�=�Dz=��q=��]�*�=x���B�J=��h��ɰ=R�=p�Y<��-=��=A0=p�k<�7>/z�����:��C<и�<�=�宽�uL=�я=�NX��"�=�r=�f��,)p���ͼ%s���=	���{U�V�<Ph6�"��� a��S �%���_K�j��=�T=�q��@�j���H���c�<mY�=���B� �U�p9߻��/�� 1<"�g���X<lT\�B�n=���;0G�&f�`������	弲8t�`�s<�
�"�J��kb����= [<�� "�x��؅<ʺu���H��6�h=��6�$���>�=8V� �,� i7<x�T<y�����=�9k��b=�=s=�_ҼP�ټp�Ǽ���=�_=A���/�<���{.=���ptż~X��8-=�h�<`QX��)�=ڎ=�ԙP���<
2�=�r�<`{;�g<wթ��ş=�I/:���<Cu<@e��hCp� dp; �~<�~�=��$=��<��ݻ��|=J�"�(KP��ؼ�J�<���:� ��8�<d�ż> J��N�;gz=hi��~(���x<x�<��u��
�=��=��=H��<�]�<hC-���= /5���V5>
==*�,��`J�p	<�?�P���p=�s|� T���z��R�]�Dӂ��� =��h<x0�<@!�;�S�����!
����<���<8��<(,ܼ�)I=��9�&E=�M= �7<|�����;@k���=$�*�3�|=r6� ׸(���!>�S=WN�='s�= �Z�`,��r]1��Ę<�S<�s��հ<	��=.��=Mcr=�-7=s�M>�T��d�_��L;<���=4������=�]���vz���&=(�����܍�<��|3=�a ��:=�OB=!o�����0e���o�]�;`�9<@��:�P�=���=t{�<Qp����� r��`�+;J��P'�;F��=�t"���<��@�-�x=�_�:�o��>�=���=�b=&��=O,L=�����;�r[=lu=͚�=z����@=�(������ =Q���(�b=��=0m�<��x=�O�=��d=��I?�=���X[�P��; �j92�<$�νTG=0�=�d��O�<�~�<�&2<�i6�p�K������^<�sp�xIi� �����;?�$�=� ��<��?�X��r�=�	=�c|���<�����G<���<3��=?#��: <`/ϼL���|U< œ<d�ln�<�� � =h�@=p��� �ݼ��?< �N�x�q��4)����<����6� ޺㣴=��z;��� ��;0�;~Lf�P�1����D7˼.m���=a-�=����u��� ���>�w�����,=��?<';i=��=��C��7��X�;��=7V-= ��� 2�9 �+��#=�ʽd=ͼ��H1<@��P���-�=jMG�J����<��=�'=@�=0!��'M����<(E:�d�I=З<䐱<�O�䉺<�y/<V�>DH�=��<�ρ<h�= � ����p����!=p�%��h��d<ּBl�8��=eW_=P`ۻ�$ ����;&k=�TʼtU�=o)�=9b=Ђ8<l��<\�ż�5�=Bͼ�Q���>P��<&�Q�Ԇ!�p���U�b|ǽ� C=p�J� �C<Z��#���4d���u@=@��< Oǹ������5��z��Ɯ�==�"��Y�p���r�<ߜ!�q�=��=�(<������Hѻ.\�=�߫�5�_=���؉�������=�s%=�{�=�Ō=���`vF�HV����i< ��<D*�� �;y�=d�=��
=5,=J>�V#��G���d�$�����<�4׼j �=@�j�D���@�z��2[��ψ� �[�����Q5=pѢ��[�<L�o=2�*���Wټ�o\�(�-� �:7"���"=\<b=0s�<�hK��V�� >�;\@�<�]� �N<y�v=���9�:+)��=��;��)��A�=(��=�yZ=�}�=$�=��^<^w�֮A=�*�;��=x�ԽpzX�,z�R��l��<3w��2�.=n�=x&�<��\=}��=(8=��x�'Q=�^���;`�!�����pU�;�Yͽ�ރ<�� =�ə��!�@�ٔ<�~]� j���?���#�;�����F�� 4����<����Ѽ}�<$����w׼�M�=xhw<)�����4=�Q�8l�<h>Q<\�<� \�IH=���6�<��k�<�*㺀�-�8=䀺���B�Y�=��.<(t�U=4��< �#�phɼqC(=8�<�����F=��v=h� <�����"�; r�:�x���	�$T���aμ"��� e=���=b�=��͜��G������b�G��W?= �<�b=��:��f�&<[�0jv<���=`�;���<��b�0��<��=����
�+���;�.�< �����~�a��=�q0�H"���q~<q�=���<:�q=�M��v�l�HRV���
�?�Y=�����,=h�%��;<8�<>�	>;�=��<4�=�؊=��<�����u�x�N=�q˼��\��||�L�Լ�-���=sHG= �;��=��YĻ��=��}:���=[��=P��<HB����<0tF���=
|5�FW	�ϣ-=�GT=��?W���/��2�� �#:^���h�=��*<��Q<�t �@7�� ^r��n��-��� e.<z=���=霎����@gT��^�=�����=4z��fR����< ��i腽�}0<�[U=�J>=x�k��z
=�<hV�=H�}<p��n=�ㄽ�Ղ=�ވ���y��i�f�4����o�<�v�?����=��<V	�ļ��R&=�P= ȯ8�����p�=K�>�̿�(_�<�T�<��f�_��=�m'���0=��G<���;�<0a��l$�f�V�dF=�=Ho=���������<�h��8�(<aI���R���=傁��r��H0��)ӿ���F�D3,���p�]S���.=�|��`�4�0�}�V�k�(�a��@=��=��ؓy��/!=P$n<  +��=�/=��Q����=��';������L=Y'1=t⩼�D?=`�K��=�6]=��=�D�Ь/�`�;W��=.�E��(/=��:T����@T�l5����/�̻ü�2�;$��f�e���C=��?� ��;�MȻ��;�#X������=�; <��M��\	�p<<Q��|i��:uj��O���_��J�<��&�D�e=\�r� e��,!e=6d� ���h�,=4�ü(�(�Č����&����!:=�Fջ�����4�@k�<�d��.9�d��< `q<�M�<�{;�؛�F�b/%��1V��vZ�J|c� �c���]=�=@�c�o���ݰ=�w$=���ܫ�<hK��p׃�`U�;
�|�(�������a=�	{���0��x���E�L��<�;Խ�ڽ�y�l"�<͂=�U�l�=��.�I�T=Zօ=��z��B,�i[/=�؀�ڽ�Fq�-���x�~<��غ�����=����Xļ��b%=�O��`\��;m!<�7<(𤼈�q��}W=t�Ｐq��@�����;��=ʰ=��d=@N?��c����B�ht�<��q�0����D~:��x;���;����YQ���>l����G��Իx=��="~x�Ĳ�|��<*�ս��!���i��A=�(��\��<@�<ɋ�HN���4�Ŭ���
<8�0=Q�d=u�Ͻ����X��T��=����q�=�&\�A]���<|���% ��H�<�s�<6ϯ=�f��x%L=X�<��=��<0�߻�xJ=$\¼��=N����oļ�������ţ�� 4Z;냽��=E��=�Q�<�m(��_=��=*�'=�p<B��$��=��=�̬��м B9=���;$�>��h;l��< �u9xCD<(�)=��:���׻U0Ƚ�b�=d�-= S�9�y�< K���6=�����N�<�y��	ɒ��j�=_H��T����=<���X��K��ȭN<��߽�v=�	�@t;Pp�<nw����<k��=�X�=�O��P�2��UP=�g�<�9��&�<���=����=��<Xk����=HA|=����@�E<��i< ��< �=F�=�^���)�hQH�=(����u�;�zO<�Ɨ����X��������pNS�L���Cx��W=��4��<�J��Έ;����HѼ`67�Lr����t���;P��<b%p��m"��½�q����W��Z=�Q��m�<�\��h;�85=�U4�џ��=Ъ�h�<�� �X�I�ȌR�|�<�뚻�ʆ��`m����<�4�<U�x'7<x�<���< �L���8�ż�@� ;}�lы�zd� �Ļ��=�+=`b5�����-=w=�穽T�<`�޼ȫ3�ʈ<����0����5��TJ�= eF��i��8���S�\f�<;H轛�ѽ�l}��d<��=8le��ơ=ځ8�

=���=(�)<e�B�=��o�1^佨N<�ّ�`�~<�Eٺ��0n�0%*�(��x&=@�-���ph'���<�������F����N=\C�X;��e�; �+;�q<@,�;k��=@��E���R3�4»<�.����<��<�>(<PZ�;���|v��
>x�� ec�μ�=L��<�H��@"�T�X=H�.��`�������)���뼌��<���;�e���XҼ�"�Ӕ��D�< !��z�<�P�����(����=`������=�����:��l=^Y+��w���t=�J��w� >����*�=yżh�=�����)= ��<�1W<;2�=��ݽ�������潼����䊼l�%�*!�=�>,o�< �;��>l��=l��</�=а]���&>�=PD;��k���_=�Z0=H�'>���;rZ<�n(���	<v1=���`�I;��5�G=+<=XS'��n�<�R�:��_=�_d�p�<���]꽩�k=�'���C��~i=�G� ���.��e�<`e/����=�a�P�D<Nh�=�il;7�_=m�=͍�=RƦ�D���B}D=��=�8��8����&>`�L����<%Q)=�ϼ���=	k�=�2��@Ih<�e&={=�h�=�1�=��I���v��q̼x��=�� <T�S�X��<H�<�-޽4��<�K�;d���ķ��A�����V=�6��y�< �: �2�����&��ܙ������ǈ�K+=��0<�n�p�%>�� σ;�	-�T�=��Y�`ϻ�1� ��:0�=�޴�Z���q=�L�P�<�J��P?Q�Y̼��< [R�dm漙������<(��.�X��<}	=�u<�������P�$�����5���Ý��m�����Vn�=��k=h�?�'���<4·<�S��6�=����B<5@$=}Ū��Ɵ�	ʒ�0b�=h�U<�f��|�<�C��$=8���*��jώ����:�G�<�AW��R�=�M� �t�=��C=��Z��@�<��l�m����<�6��0T�<���jM%� �I�Ȃ1��lV<�K�=��<��c�8�s�d�<��ؼ������i�K=L���<�E� �R<���@�;�6_<���=P������|TB� � =�h���q>=��{=T�<�rX:�m�;�,{��>0����W���]7>�=c	�������=��N�]
��<Z����u!� ��<ر���J��$l�L^��J�h�,��<>�.���:��&��a�8�J<߹=xѶ����=��4�x�V�d��=�q:�����e�f=Йd��<>4p�H�=��-���E=����i�=��;�JZ=v��=$���x&
�'줽P��n�N_��R��;;�=�
!>�*5=(pm<�f><��=H��>�E=�G~��4>��<�@<ر���@=�z�=.��=X<�l�<i�����<l�<�pE<�PY<�n�����<
<���f<�O�<yC=��=�d=�ڼ���꽊FI=4����8I�q��=·��5=�1��<D�M�N?W=�@>�`%1<D�=j�C=RΒ=�h>�ѩ=�Q��@<�=��=d2�<Hig��&>Џz�H�5���?=
�4��ܹ=�J�=`���.=�}=X�>=e��=���=o���Lɠ�dN���pY=]W#=J˽���<�E�=��ǽ,��< <��ɼ�	�8���`½z�(=��U�@��: O�;Xkn<��ڽ�(�@��0`,�h|h�6�v= �H<�@��h�
�3jѽ�Ǥ;����9��=R�ʀ<lY ��:���,=pw�;���Ԩ=̢W��x<�M����R��\����< �ѹ��������(�<�y��R��(�6<!~k=��<�ݮ��T������	��.����.u�{���d� �=>Ʒ=�[���{��_�<��<)i��)�N=��;`�<i�F=I,��R���3O�\�=8�b<PNS�@��;���,�=o&��gi��`�D����;����X����=R�k�pb�H�l=��e=�3P<�=����߾ὰ��<kU���k�<pI������ �;�������<��=��= N�;��;o�l=P�M�@�;��g}���X=�$��\�|��	Ż �^��u;�=��=��u;�Q�t� ��S=��T��=:Y�=��=�"���G<��7��B>�ܼ�����>>L�5=�C�� ��=X?:��i�x��<f��R����;zC��o��p�X<�{��6��@�z;m����������뇼�=<��}=ܪ���k�=m�@��=?ˇ=>i �L��0�;܌a��� >8���j�=@�м��h<�V:����=�n|<1��=���=�.����ü*'�p��; ���:5G���I;_��=�$>v�n=�i�<�Ln>d�+=��Z�T�0=`�}��>�Z����\=\�(��=κi=��= �?;�O�<v��,�R=@'�;n�=��<͊�l]x�8<6R��P���@W�<D� =+�8=�^R=���b�� �=�n������=��M;.3�=��'�D��<x�@�49=����`]<t��=�+�==�=]��=��h=l������<& =[�=H�V=�ܣ�k�=�I���7ӽh��<�qp����=�=�t�<
^= ��=�3a=D��<%��=�\��Mh�𪝼��=<�=8F�;�<�JR=�����P�� U��į��T���Ȃ������<�y�x6���2<Ԟ=�sj�<�� �Y<���;�H���=���<C���<1���0����|8�GD=� ��@t=��8��[��^=�<�'�F� =<�$���_�̼�<��+��qƼp@=�j<�{��2��$�<ػܼ�^�b�=�e�=`q<��2��V��p����N�4�������]J� Aͻ�ܘ=���=*����齼��<�m�<͗����=��<:�<���<�.��^?1�t炼�r�=p�[� .o� ��������X=Kö�?��Xn�|�<��B���\��<�=��t�`��;�� =�!\=��<ўI=uڈ��Է�` ��Qc���}= ��>5=��9����;@N�6�>聋=�֋<d�=ݕ=��<��0���!�?w=|����\����8�<@2<<ٴ�=�Ы=�2�<F���mڼϒ�=:�0����=��=l�=�̼��<HM����>^�D����\>��<��t�����U+����3����P=�TY�𵒼�o(���������+�<`���j�6��r���k�H�漱����j<�H����<�<j��;=o &��J�=��==�~�/ۋ�
�d�<bü�5�=��v��=@B*���v�X[�Π�=��<ң�=mw�=�>�`򥼰]�[�; �!<��e�ػ����=k
�=\Z�=��=�iB>�l�ppG�`g<P$h����=��޼g�t=��t� ����;s��T�� ��f��)�_= ���3�<Ym@=��������`����X��ڋ�@�r<P�<��Q<��b= j��$�����;�M�� �̹|)������s0=UD� ޻R�(��ω<`A7;�,�<�cP=l"P=�M�=��y=�}=��<PǴ���G=О�<�7}=R�ý z?<�Gн4$罀��;�����c=b=b=�x=)SR=`��=��9=�W޼�V%=������C����pf�; i��P��լ<@k�;�a�*�o���V��һL�<\�<�a���#M;Hda��޼�4
<��/=
��8.����^<�c�<hV��}Hm=+<]���8��<�� �@+=��+@���v<���;*�=Е0��*i���I=��;�[�;�[ =$ټ���G=����`5$��:j=t�=��<<����!=�O�,���8ht=�X=�T�; �/�@�̺ l�88���H���������ʻ�,=���=z�L�I[��qԻ@�!��M�>W~=6� =|ؤ<�Ɓ��œ�F�5���+<js�=\�K=��� �< =f�c����� �<��=@�#�����s�=N�K��X�;Я<�j=`�<��z=�F���jc��郼�#�)=�,���s_=�`����:@}@;_l>n��=D��</�Z=�Ʉ=�5�<�l�TK�����=�W��Cɼ$Vi���;��<zD>\��=��<�ͼp~^�U�=�V��c=4=t�<��5���O< ��;R��=��x�lw���#=Di=C��d�
��qԼ �༐��;�l��E�=��U<�m< ͟�0��;�G�� �;�ɳ���<��K=u�x=���*���.���e=�W+�٫=�s��(ꄼ	=�
����(���ļ��E=��F=|v"�X�<.\=YGB=@��;�p��Ù=c��\g="t� JQ��񗼻������8��<VC��H�ټD�=􀋼ث>���8�`�C<(�2<��;p����G=��
>�4���W=؃*�,M༕��= ��<*1a=��'<��;�q�<�� �޼����p=��<Wm=��{�*1�tw�<Z{����;��~�~�.��]�<f[#�l������9ѫ�|�g�x�F<<�ܼ�3��x<@��� �:̿��N3`���o�P�=�k=JL�ƅ����=��� /Ż�=h�%< pU;d�=��׼h�u���=��<8�)��X�= W���r>=�Y?=?�<H����»�C����=2Wp��D=T̻���0?/��8���G�``i����<�������ai<=�R����:$���)(���U��&*��<.@���ͦ��߻�FȽ�qm=Ɨ���,=
D�@9<Z��R�=� ��;��
=�Q"���|�r�_=�D�)�<iK<�������;@��;6� =�\��N�v��Ë<�u��U���hN=P�;<�G; ���r��l���U˽��Y���6�3ɮ�2�D�q�=�i=�;!��f��g<D����j���'<Ȕ�<��<h(�<_潽��C�r�O��e=@��<HM���t����a< u�<~���������< ����j<(p��pa�=��c��?�:��= ���� ����=����c�׽��c�����s�<�W��Tz�����;ط�`n�;�=ϼ@�J;Hs��W�<U�=��w��@��Vx=�*�� m��&����z��=+�=�fD=ಚ�G
���C��(�,=�R���4=���;�C����;/�!=`�ּa >H�������B=f=߈��p0ͼ�,X��S������89����D=8�D<��<��λ$�$[��pS������o�P��<��M=
iN��	��Zڼ�ϰ=�_����=�m1��.�9=�@�?�������<�0�=��j�<:=pJ�<�va=Pw��cP��"=L�޼�=�ս0p���A��3������@��:�Jн��=��=�q�,�� �;��w=z�-=�y�:���:�(�=R7�=�*�� �-9���<�����>���잍<0�;�a8��h9=,A����
;#t���K�=�f�<@�:x�<TFӼE�D=0C���刻�F��%���E�<"�2��@��`Ђ�!n���ו���"�@h�;Q?Խ"�<���{<P��<�-W�P̡<ˉy=
��=L����y����C=���(Yü�&�;B��=p����=p��� ��V�=�@I=�yؼ��<��u<�V5=�Ó=x�<p�:�$�Ƽ�!���=U��"j<���H�ͼ�A����A<����t���;�ӧ��[��&�F=G�`�滤
!�b�wɽHVo�H8���'�2�Ƚ���0��;�L���u9=��z�]�/=p����<�(� �';�H𼀥I<H��<0턼<����4Q=.��h=�G<�6ɼ���;ؖ����<f���՞���<Њ��x伖-=Ò< �	:��+��L�� �Z�))ڽ�p���P��x���n�*B�=. I=�CK�GZ콨˖<<����.��zi< HN9,��<L�<i�ɽv&��R�֔�=8��<���0�;���;�r�< [�5P����Һ(3�����;`�G�9&�=�:��pj��X��=ԙ�< %����=?n����޽������0"�<�O��0����;l*�$��<���=��e���2<t�/����<�$�<�̃�����켂=�Ū�N|��8T?� �-���D=az=|}=�ż���j���2=�K�c��=���<��p=�;9dR=�׼j�>@/7� �S�8o�=���<߯���}��@=oz������v <�o)��S��<�Y �6�R������}� ���
m��B'<7]ѽ^Lʽ���l0�=d�ּ$�>7��ޜ �:)�=x�����N=ܑ���T�=$�p��3c=($��V��=�u����w; Z�X2�<��=k��B�ۗ��Z%x�����ټ�[�����=Vm!>��!�����f>���=�=P�7<���<�X�=1z=ą-��d��j�<�ى<.+<>�����`�Z,4�r��@�=\���XƲ<������m=�N0=�l�xnP<�g���]=�R���4�+�H� ��Y�<l�����ڼ\�g=�5�,�^�,��?M=��K�d4�<�p��<�<4��=X>`��#�=8�=n��=β���X����<�t<4�߼X�]��� >(����V$=x�<R��ۤ�=���=|&��Dܥ<�~=0
�=���=[p=֊Q��F,�������=td���ʋ� �9�~<�p	�L��<��):`��� B������EԽZ;=b�@���:(O��S��?�ڽ(u��D
Լ�<��9Ľ0G<<��$<��
�=�d���$=ES��Q=�\)�l���8|̼���; F�<��;�/&���=�K+�X��<h<�x� c���<�p5�<&�,�}x����<$�%��)�TT�<�Ҷ<@�2��#���� ��q�ս|!��<q�U~��R�j����=pj=�0i������<�椼[��~�< �0���=��
=�{ӽFn��A�,{�=Б�<���N�<��<;TU�<\h�V���1X�Ȯ�� ls:�m�����=T۰��G�hW|=�R=@EI;R2�=�+���E�p����R��(y�<�Ƽ�� ���_%��=o�=Q�;x�\<����T= z:;&�������?��= �Լ�㊽P/׻��*�@"=��=Ȝ=���mᑽ�?��Ga=^6H�aY�=�M=`��;��dh=4����>\3�pj��|�/>�c%=y⿽T@��Hh�=�?�dz��)�<�w̽�����W< ,�8R����=�y���@{�:� j�psa��e������rٻ���=�n�J+�=�6��u��b>�=Ş��1����=6w��>��!��=� >��#�=����&@=z�����=7�=ϐ�\�O�#w���)K�"�a���H�84�`�>�3>4D���<��d�b>��=�x�;v�=ľ=]y*>`�:; c�;�:��=׍R=�&> ��7@�M��㘽� a���{<���8��<�T�Pڦ;�!Q=v��� �;��<��9=d�< ѷ:���T�T�=p������`�=xŰ��
�9�A��f@=��f��a�<�f���!�<��=P_�<��=��={��=�퉽��e�@��;ʆf=0�`�Ju���.>>R���h+$�@X�<�]����=6��=�ނ�:�=��I=�B�=��=�@�=�Ȩ�2�>�P�i��d=��<�� ��;�hM=�� ��.��T��
�8�j�x��IQ��6�
=������;� 1��n�;n�a�ȏi�4ꑼ�x<�o�,��<��<���0*P<����`=;�]���=�CѼ ��<����*ռ�3=(�<X�� ��<�^%���|���<��p��&r�b�=Pi�<l�b�U�\�<��:J�f���<��<�U��`g>�&(]� �#:-ؐ�X�|��*�����⮼�ǳ=,�=	<���ٽ��=�<�l��'=� <0g<L�<%�ý0���v�"��߱=�ș�(s��P��;н��T�<��۽w{˽�6v�p��@�L���=����=���w;{?=��/=��;��v=M���5,ν�񸼭����<�����\<Xx���ü�&<��>���<0�<��;b�U=�)<��ru��%u�=|��6�D#���A�Q4=�d�=8��=�K<��,���4���=�3+��ĝ=���<x�<2��h��<)]����=,x���B����;>A�F=����)��m�=<�.�n*!��=�L�EB���><��μ��x�<�T
���T��`k���﬽4ȼh� �,�c�����%��=`���Z^�==�D� f�:mU�=' ���Ჽ@%I=�d��%��= E��<�=����#=᪽���=���1��=G}�=��нğ�*�h�X߼���@�������>��%>�Y\< 3��!ހ>���=(���:=�A�<��+>V�$�	�U=P�"���<��z=��=�|�<�(<�g�P��< �b; �3<p��<�����:2=ny����$��=0��<���<0��<�*��3��^= np;�H%��^�=��;p͖<C�;��=�]G�`�;jgn��}=L��=��1=c��=�)�=E��=D� �@�< �����=Xru<V׽��>Z�^҈��K�<�M8��v�=�w�=���<�_=��Y=�[|=m��=��=��ؽ�A���X��3=NF=b/��7T<M33=��ƽjy/�(2
�
 !������l<e����8�<07��p�;��F< l=Г�;P郻mٻ�<ج�~J=�2=_�����q;��8�Tu�xW��w�<�֏��l_=4{�r?d�]�G=�0<�����;��`�:���V<�z���-Y��؏=8��<i��8�� c�< _�<�g�W�<�m�<����G�<X߼@(6;,�ڼ��D�xMk�ƖT�(N<�i<=���=
U����0�=Xs�<�V�ޒW=��<X6�@�/;����,̈́���Ƽ���=��{�к:< Թ� ���,�=�+k���ý0�v<��<�X廄�m�8+W=��=��8�<��<X6'=��<��}=O ���-���-������?�<>¼#�Y=_R��挻�iܻ+�>,B;=lˁ<c&=5�j=���<�����D���=Zw.�@Ê��Q�\��<��A=��=���=��<�A���G��ʃ�=��ZqY=�t<<eF<N�a�@]� 0K����=�P����~�%>"�=		��X6p���=j��~��ߦ,=��Ľ�%��+0;fp����(i<4����ڇ���{��������������^'���j=��2��1P=�-��a=��=�E��������r�|7}����=0㱻��=�چ���f:�匽��d=D�����=���=,;��H���xm���4� �k��웽X<��.��=ӥ�=b�V=@R�:��`>���<��G���<�@�;� >n�H�2Q|=���PS<��<p��<L4�< �';����D��< �Ṹ"�<@J�<3�ݽL�|��j�<"����¹���<䠅<@Lm;jE-=���١�=�-=@.ڻ�h���<��;�K�<8�XF<�O$� ^�|����(=p�= +=�ݴ=旃=c^W=  F9��;��Y<a�=w0=�ҽoq=Rg����ƽ@q�;"ad�� `=�3=�2=�=�!�=�H=W�;=Ia6=��������k���<�E�<r
��β<���:��}��k��\�e��m�l��<1�'=��P���; �U;�1�;|��<b�*=o�J=h~!<Pi��R=�}���<�=��H���<B��@�4���?�46�<�R�=�r��j`�!�#= xQ;�n< Е���ܼ�bW�,B�<X�?� O���Q�=��<�*Y<x,�<���<�� =�������<P��;�+��ȡ1<�$���<�?K<�ۼ  /<�t̼3<�.<`�="�/���A�@�J;Д���>&��tn=FB=�T+�H2����C�L}�� ��:��B=dS����
=�)޼�[<��=�~�������=�= (���!�H�<����i�<$h���f8=0?<��m=嗡���뼘���A|��ײ<�u��(q=��"��u��������=O[=�f�<��t=�4C=���<XAT<�t����X=�!.�=d<�oX�x��<�)=�4�=�^�=�{=��9� �
;��=��<��|�<����0�;-���@w�����<�Œ=v}t��S�;*
dtype0*'
_output_shapes
:�
�
siamese_3/scala1/Conv2DConv2DPlaceholder_2 siamese/scala1/conv/weights/read*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:{{`
�
siamese_3/scala1/AddAddsiamese_3/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:{{`
�
/siamese_3/scala1/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala1/moments/meanMeansiamese_3/scala1/Add/siamese_3/scala1/moments/mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
%siamese_3/scala1/moments/StopGradientStopGradientsiamese_3/scala1/moments/mean*&
_output_shapes
:`*
T0
�
*siamese_3/scala1/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala1/Add%siamese_3/scala1/moments/StopGradient*
T0*&
_output_shapes
:{{`
�
3siamese_3/scala1/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_3/scala1/moments/varianceMean*siamese_3/scala1/moments/SquaredDifference3siamese_3/scala1/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
 siamese_3/scala1/moments/SqueezeSqueezesiamese_3/scala1/moments/mean*
_output_shapes
:`*
squeeze_dims
 *
T0
�
"siamese_3/scala1/moments/Squeeze_1Squeeze!siamese_3/scala1/moments/variance*
_output_shapes
:`*
squeeze_dims
 *
T0
�
&siamese_3/scala1/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/zerosConst*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    *
dtype0
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read siamese_3/scala1/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_3/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Nsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Csiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_3/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Esiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstI^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAddl^siamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Dsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_3/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_3/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
(siamese_3/scala1/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *RI�9
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*    *
dtype0*
_output_shapes
:`
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read"siamese_3/scala1/moments/Squeeze_1*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_3/scala1/AssignMovingAvg_1/decay*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
usiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Tsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Isiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_3/scala1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstO^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAddv^siamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *  �?
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
"siamese_3/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_3/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
c
siamese_3/scala1/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_3/scala1/cond/switch_tIdentitysiamese_3/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_3/scala1/cond/switch_fIdentitysiamese_3/scala1/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_3/scala1/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala1/cond/Switch_1Switch siamese_3/scala1/moments/Squeezesiamese_3/scala1/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala1/moments/Squeeze* 
_output_shapes
:`:`
�
siamese_3/scala1/cond/Switch_2Switch"siamese_3/scala1/moments/Squeeze_1siamese_3/scala1/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala1/moments/Squeeze_1* 
_output_shapes
:`:`
�
siamese_3/scala1/cond/Switch_3Switch"siamese/scala1/bn/moving_mean/readsiamese_3/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_3/scala1/cond/Switch_4Switch&siamese/scala1/bn/moving_variance/readsiamese_3/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
siamese_3/scala1/cond/MergeMergesiamese_3/scala1/cond/Switch_3 siamese_3/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese_3/scala1/cond/Merge_1Mergesiamese_3/scala1/cond/Switch_4 siamese_3/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese_3/scala1/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese_3/scala1/batchnorm/addAddsiamese_3/scala1/cond/Merge_1 siamese_3/scala1/batchnorm/add/y*
_output_shapes
:`*
T0
n
 siamese_3/scala1/batchnorm/RsqrtRsqrtsiamese_3/scala1/batchnorm/add*
T0*
_output_shapes
:`
�
siamese_3/scala1/batchnorm/mulMul siamese_3/scala1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
�
 siamese_3/scala1/batchnorm/mul_1Mulsiamese_3/scala1/Addsiamese_3/scala1/batchnorm/mul*&
_output_shapes
:{{`*
T0
�
 siamese_3/scala1/batchnorm/mul_2Mulsiamese_3/scala1/cond/Mergesiamese_3/scala1/batchnorm/mul*
_output_shapes
:`*
T0
�
siamese_3/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_3/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese_3/scala1/batchnorm/add_1Add siamese_3/scala1/batchnorm/mul_1siamese_3/scala1/batchnorm/sub*&
_output_shapes
:{{`*
T0
p
siamese_3/scala1/ReluRelu siamese_3/scala1/batchnorm/add_1*
T0*&
_output_shapes
:{{`
�
siamese_3/scala1/poll/MaxPoolMaxPoolsiamese_3/scala1/Relu*&
_output_shapes
:==`*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
X
siamese_3/scala2/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese_3/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/splitSplit siamese_3/scala2/split/split_dimsiamese_3/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:==0:==0*
	num_split
Z
siamese_3/scala2/Const_1Const*
dtype0*
_output_shapes
: *
value	B :
d
"siamese_3/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/split_1Split"siamese_3/scala2/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
�
siamese_3/scala2/Conv2DConv2Dsiamese_3/scala2/splitsiamese_3/scala2/split_1*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�
�
siamese_3/scala2/Conv2D_1Conv2Dsiamese_3/scala2/split:1siamese_3/scala2/split_1:1*'
_output_shapes
:99�*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_3/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/concatConcatV2siamese_3/scala2/Conv2Dsiamese_3/scala2/Conv2D_1siamese_3/scala2/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:99�
�
siamese_3/scala2/AddAddsiamese_3/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:99�
�
/siamese_3/scala2/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala2/moments/meanMeansiamese_3/scala2/Add/siamese_3/scala2/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_3/scala2/moments/StopGradientStopGradientsiamese_3/scala2/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_3/scala2/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala2/Add%siamese_3/scala2/moments/StopGradient*
T0*'
_output_shapes
:99�
�
3siamese_3/scala2/moments/variance/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
!siamese_3/scala2/moments/varianceMean*siamese_3/scala2/moments/SquaredDifference3siamese_3/scala2/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_3/scala2/moments/SqueezeSqueezesiamese_3/scala2/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_3/scala2/moments/Squeeze_1Squeeze!siamese_3/scala2/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_3/scala2/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/zerosConst*
dtype0*
_output_shapes	
:�*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB�*    
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read siamese_3/scala2/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_3/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
ksiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?
�
Hsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_3/scala2/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstI^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAddl^siamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese_3/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_3/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
(siamese_3/scala2/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/zerosConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read"siamese_3/scala2/moments/Squeeze_1*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub(siamese_3/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
usiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_3/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstO^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAddv^siamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
T0
�
Jsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
"siamese_3/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_3/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
c
siamese_3/scala2/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_3/scala2/cond/switch_tIdentitysiamese_3/scala2/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_3/scala2/cond/switch_fIdentitysiamese_3/scala2/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_3/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala2/cond/Switch_1Switch siamese_3/scala2/moments/Squeezesiamese_3/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*3
_class)
'%loc:@siamese_3/scala2/moments/Squeeze
�
siamese_3/scala2/cond/Switch_2Switch"siamese_3/scala2/moments/Squeeze_1siamese_3/scala2/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala2/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_3/scala2/cond/Switch_3Switch"siamese/scala2/bn/moving_mean/readsiamese_3/scala2/cond/pred_id*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese_3/scala2/cond/Switch_4Switch&siamese/scala2/bn/moving_variance/readsiamese_3/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_3/scala2/cond/MergeMergesiamese_3/scala2/cond/Switch_3 siamese_3/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_3/scala2/cond/Merge_1Mergesiamese_3/scala2/cond/Switch_4 siamese_3/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_3/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_3/scala2/batchnorm/addAddsiamese_3/scala2/cond/Merge_1 siamese_3/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_3/scala2/batchnorm/RsqrtRsqrtsiamese_3/scala2/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_3/scala2/batchnorm/mulMul siamese_3/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_3/scala2/batchnorm/mul_1Mulsiamese_3/scala2/Addsiamese_3/scala2/batchnorm/mul*'
_output_shapes
:99�*
T0
�
 siamese_3/scala2/batchnorm/mul_2Mulsiamese_3/scala2/cond/Mergesiamese_3/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_3/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_3/scala2/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_3/scala2/batchnorm/add_1Add siamese_3/scala2/batchnorm/mul_1siamese_3/scala2/batchnorm/sub*'
_output_shapes
:99�*
T0
q
siamese_3/scala2/ReluRelu siamese_3/scala2/batchnorm/add_1*
T0*'
_output_shapes
:99�
�
siamese_3/scala2/poll/MaxPoolMaxPoolsiamese_3/scala2/Relu*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID*'
_output_shapes
:�
�
siamese_3/scala3/Conv2DConv2Dsiamese_3/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese_3/scala3/AddAddsiamese_3/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_3/scala3/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala3/moments/meanMeansiamese_3/scala3/Add/siamese_3/scala3/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_3/scala3/moments/StopGradientStopGradientsiamese_3/scala3/moments/mean*'
_output_shapes
:�*
T0
�
*siamese_3/scala3/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala3/Add%siamese_3/scala3/moments/StopGradient*
T0*'
_output_shapes
:�
�
3siamese_3/scala3/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_3/scala3/moments/varianceMean*siamese_3/scala3/moments/SquaredDifference3siamese_3/scala3/moments/variance/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
 siamese_3/scala3/moments/SqueezeSqueezesiamese_3/scala3/moments/mean*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
"siamese_3/scala3/moments/Squeeze_1Squeeze!siamese_3/scala3/moments/variance*
_output_shapes	
:�*
squeeze_dims
 *
T0
�
&siamese_3/scala3/AssignMovingAvg/decayConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *RI�9
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read siamese_3/scala3/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_3/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_3/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstI^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAddl^siamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
 siamese_3/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_3/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
(siamese_3/scala3/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *RI�9
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read"siamese_3/scala3/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_3/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Nsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_3/scala3/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
T0
�
Hsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstO^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAddv^siamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese_3/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_3/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
c
siamese_3/scala3/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_3/scala3/cond/switch_tIdentitysiamese_3/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_3/scala3/cond/switch_fIdentitysiamese_3/scala3/cond/Switch*
_output_shapes
: *
T0

W
siamese_3/scala3/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_3/scala3/cond/Switch_1Switch siamese_3/scala3/moments/Squeezesiamese_3/scala3/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala3/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_3/scala3/cond/Switch_2Switch"siamese_3/scala3/moments/Squeeze_1siamese_3/scala3/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala3/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_3/scala3/cond/Switch_3Switch"siamese/scala3/bn/moving_mean/readsiamese_3/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
siamese_3/scala3/cond/Switch_4Switch&siamese/scala3/bn/moving_variance/readsiamese_3/scala3/cond/pred_id*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese_3/scala3/cond/MergeMergesiamese_3/scala3/cond/Switch_3 siamese_3/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_3/scala3/cond/Merge_1Mergesiamese_3/scala3/cond/Switch_4 siamese_3/scala3/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_3/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_3/scala3/batchnorm/addAddsiamese_3/scala3/cond/Merge_1 siamese_3/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_3/scala3/batchnorm/RsqrtRsqrtsiamese_3/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_3/scala3/batchnorm/mulMul siamese_3/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_3/scala3/batchnorm/mul_1Mulsiamese_3/scala3/Addsiamese_3/scala3/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_3/scala3/batchnorm/mul_2Mulsiamese_3/scala3/cond/Mergesiamese_3/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_3/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_3/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_3/scala3/batchnorm/add_1Add siamese_3/scala3/batchnorm/mul_1siamese_3/scala3/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_3/scala3/ReluRelu siamese_3/scala3/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_3/scala4/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese_3/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/splitSplit siamese_3/scala4/split/split_dimsiamese_3/scala3/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
Z
siamese_3/scala4/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_3/scala4/split_1/split_dimConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese_3/scala4/split_1Split"siamese_3/scala4/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_3/scala4/Conv2DConv2Dsiamese_3/scala4/splitsiamese_3/scala4/split_1*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese_3/scala4/Conv2D_1Conv2Dsiamese_3/scala4/split:1siamese_3/scala4/split_1:1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
^
siamese_3/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/concatConcatV2siamese_3/scala4/Conv2Dsiamese_3/scala4/Conv2D_1siamese_3/scala4/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_3/scala4/AddAddsiamese_3/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_3/scala4/moments/mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_3/scala4/moments/meanMeansiamese_3/scala4/Add/siamese_3/scala4/moments/mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_3/scala4/moments/StopGradientStopGradientsiamese_3/scala4/moments/mean*
T0*'
_output_shapes
:�
�
*siamese_3/scala4/moments/SquaredDifferenceSquaredDifferencesiamese_3/scala4/Add%siamese_3/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
3siamese_3/scala4/moments/variance/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
!siamese_3/scala4/moments/varianceMean*siamese_3/scala4/moments/SquaredDifference3siamese_3/scala4/moments/variance/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
 siamese_3/scala4/moments/SqueezeSqueezesiamese_3/scala4/moments/mean*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
"siamese_3/scala4/moments/Squeeze_1Squeeze!siamese_3/scala4/moments/variance*
squeeze_dims
 *
T0*
_output_shapes	
:�
�
&siamese_3/scala4/AssignMovingAvg/decayConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/zerosConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read siamese_3/scala4/moments/Squeeze*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_3/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
Nsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Hsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Csiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_3/scala4/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstI^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAddl^siamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
 siamese_3/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_3/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese_3/scala4/AssignMovingAvg_1/decayConst*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/zerosConst*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*    *
dtype0
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read"siamese_3/scala4/moments/Squeeze_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_3/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
usiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0
�
Nsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Isiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_3/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstO^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAddv^siamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Jsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese_3/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_3/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
c
siamese_3/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_3/scala4/cond/switch_tIdentitysiamese_3/scala4/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_3/scala4/cond/switch_fIdentitysiamese_3/scala4/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_3/scala4/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_3/scala4/cond/Switch_1Switch siamese_3/scala4/moments/Squeezesiamese_3/scala4/cond/pred_id*
T0*3
_class)
'%loc:@siamese_3/scala4/moments/Squeeze*"
_output_shapes
:�:�
�
siamese_3/scala4/cond/Switch_2Switch"siamese_3/scala4/moments/Squeeze_1siamese_3/scala4/cond/pred_id*
T0*5
_class+
)'loc:@siamese_3/scala4/moments/Squeeze_1*"
_output_shapes
:�:�
�
siamese_3/scala4/cond/Switch_3Switch"siamese/scala4/bn/moving_mean/readsiamese_3/scala4/cond/pred_id*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese_3/scala4/cond/Switch_4Switch&siamese/scala4/bn/moving_variance/readsiamese_3/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese_3/scala4/cond/MergeMergesiamese_3/scala4/cond/Switch_3 siamese_3/scala4/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_3/scala4/cond/Merge_1Mergesiamese_3/scala4/cond/Switch_4 siamese_3/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_3/scala4/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_3/scala4/batchnorm/addAddsiamese_3/scala4/cond/Merge_1 siamese_3/scala4/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_3/scala4/batchnorm/RsqrtRsqrtsiamese_3/scala4/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_3/scala4/batchnorm/mulMul siamese_3/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_3/scala4/batchnorm/mul_1Mulsiamese_3/scala4/Addsiamese_3/scala4/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_3/scala4/batchnorm/mul_2Mulsiamese_3/scala4/cond/Mergesiamese_3/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_3/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_3/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_3/scala4/batchnorm/add_1Add siamese_3/scala4/batchnorm/mul_1siamese_3/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_3/scala4/ReluRelu siamese_3/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
X
siamese_3/scala5/ConstConst*
dtype0*
_output_shapes
: *
value	B :
b
 siamese_3/scala5/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_3/scala5/splitSplit siamese_3/scala5/split/split_dimsiamese_3/scala4/Relu*:
_output_shapes(
&:�:�*
	num_split*
T0
Z
siamese_3/scala5/Const_1Const*
value	B :*
dtype0*
_output_shapes
: 
d
"siamese_3/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala5/split_1Split"siamese_3/scala5/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese_3/scala5/Conv2DConv2Dsiamese_3/scala5/splitsiamese_3/scala5/split_1*
paddingVALID*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
�
siamese_3/scala5/Conv2D_1Conv2Dsiamese_3/scala5/split:1siamese_3/scala5/split_1:1*'
_output_shapes
:�*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_3/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_3/scala5/concatConcatV2siamese_3/scala5/Conv2Dsiamese_3/scala5/Conv2D_1siamese_3/scala5/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
�
siamese_3/scala5/AddAddsiamese_3/scala5/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
O
score_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :
Y
score_1/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_1/splitSplitscore_1/split/split_dimsiamese_3/scala5/Add*
T0*M
_output_shapes;
9:�:�:�*
	num_split
�
score_1/Conv2DConv2Dscore_1/splitConst*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score_1/Conv2D_1Conv2Dscore_1/split:1Const*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score_1/Conv2D_2Conv2Dscore_1/split:2Const*
paddingVALID*&
_output_shapes
:*
	dilations
*
T0*
strides
*
data_formatNHWC*
use_cudnn_on_gpu(
U
score_1/concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
�
score_1/concatConcatV2score_1/Conv2Dscore_1/Conv2D_1score_1/Conv2D_2score_1/concat/axis*
N*&
_output_shapes
:*

Tidx0*
T0
�
adjust_1/Conv2DConv2Dscore_1/concatadjust/weights/read*&
_output_shapes
:*
	dilations
*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
i
adjust_1/AddAddadjust_1/Conv2Dadjust/biases/read*&
_output_shapes
:*
T0"�Ɓ�