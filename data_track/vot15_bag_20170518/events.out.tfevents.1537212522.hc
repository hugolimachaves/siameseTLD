       �K"	  � ��Abrain.Event:2����D     O�K2	ԅ� ��A"��
l
PlaceholderPlaceholder*&
_output_shapes
:*
shape:*
dtype0
r
Placeholder_1Placeholder*
shape:��*
dtype0*(
_output_shapes
:��
n
Placeholder_2Placeholder*
dtype0*&
_output_shapes
:*
shape:
r
Placeholder_3Placeholder*
dtype0*(
_output_shapes
:��*
shape:��
M
is_trainingConst*
value	B
 Z *
dtype0
*
_output_shapes
: 
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
?siamese/scala1/conv/weights/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@siamese/scala1/conv/weights*
valueB
 *���<
�
Hsiamese/scala1/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala1/conv/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*&
_output_shapes
:`*

seed *
T0*.
_class$
" loc:@siamese/scala1/conv/weights
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
VariableV2*
	container *
shape:`*
dtype0*&
_output_shapes
:`*
shared_name *.
_class$
" loc:@siamese/scala1/conv/weights
�
"siamese/scala1/conv/weights/AssignAssignsiamese/scala1/conv/weights8siamese/scala1/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`
�
 siamese/scala1/conv/weights/readIdentitysiamese/scala1/conv/weights*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*&
_output_shapes
:`
�
<siamese/scala1/conv/weights/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *o:*.
_class$
" loc:@siamese/scala1/conv/weights*
dtype0
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
,siamese/scala1/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala1/conv/biases*
valueB`*���=*
dtype0*
_output_shapes
:`
�
siamese/scala1/conv/biases
VariableV2*
dtype0*
_output_shapes
:`*
shared_name *-
_class#
!loc:@siamese/scala1/conv/biases*
	container *
shape:`
�
!siamese/scala1/conv/biases/AssignAssignsiamese/scala1/conv/biases,siamese/scala1/conv/biases/Initializer/Const*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`*
use_locking(
�
siamese/scala1/conv/biases/readIdentitysiamese/scala1/conv/biases*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
_output_shapes
:`
�
siamese/scala1/Conv2DConv2DPlaceholder_2 siamese/scala1/conv/weights/read*&
_output_shapes
:;;`*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala1/AddAddsiamese/scala1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:;;`*
T0
�
(siamese/scala1/bn/beta/Initializer/ConstConst*
dtype0*
_output_shapes
:`*)
_class
loc:@siamese/scala1/bn/beta*
valueB`*    
�
siamese/scala1/bn/beta
VariableV2*
shape:`*
dtype0*
_output_shapes
:`*
shared_name *)
_class
loc:@siamese/scala1/bn/beta*
	container 
�
siamese/scala1/bn/beta/AssignAssignsiamese/scala1/bn/beta(siamese/scala1/bn/beta/Initializer/Const*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
�
siamese/scala1/bn/beta/readIdentitysiamese/scala1/bn/beta*
T0*)
_class
loc:@siamese/scala1/bn/beta*
_output_shapes
:`
�
)siamese/scala1/bn/gamma/Initializer/ConstConst**
_class 
loc:@siamese/scala1/bn/gamma*
valueB`*  �?*
dtype0*
_output_shapes
:`
�
siamese/scala1/bn/gamma
VariableV2*
shape:`*
dtype0*
_output_shapes
:`*
shared_name **
_class 
loc:@siamese/scala1/bn/gamma*
	container 
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
/siamese/scala1/bn/moving_mean/Initializer/ConstConst*
dtype0*
_output_shapes
:`*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB`*    
�
siamese/scala1/bn/moving_mean
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape:`*
dtype0*
_output_shapes
:`
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
"siamese/scala1/bn/moving_mean/readIdentitysiamese/scala1/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
3siamese/scala1/bn/moving_variance/Initializer/ConstConst*
_output_shapes
:`*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB`*  �?*
dtype0
�
!siamese/scala1/bn/moving_variance
VariableV2*
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape:`*
dtype0*
_output_shapes
:`
�
(siamese/scala1/bn/moving_variance/AssignAssign!siamese/scala1/bn/moving_variance3siamese/scala1/bn/moving_variance/Initializer/Const*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`*
use_locking(
�
&siamese/scala1/bn/moving_variance/readIdentity!siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
-siamese/scala1/moments/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese/scala1/moments/MeanMeansiamese/scala1/Add-siamese/scala1/moments/Mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
#siamese/scala1/moments/StopGradientStopGradientsiamese/scala1/moments/Mean*
T0*&
_output_shapes
:`
w
2siamese/scala1/moments/sufficient_statistics/ConstConst*
valueB
 * ��F*
dtype0*
_output_shapes
: 
�
0siamese/scala1/moments/sufficient_statistics/SubSubsiamese/scala1/Add#siamese/scala1/moments/StopGradient*&
_output_shapes
:;;`*
T0
�
>siamese/scala1/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese/scala1/Add#siamese/scala1/moments/StopGradient*&
_output_shapes
:;;`*
T0
�
Fsiamese/scala1/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
4siamese/scala1/moments/sufficient_statistics/mean_ssSum0siamese/scala1/moments/sufficient_statistics/SubFsiamese/scala1/moments/sufficient_statistics/mean_ss/reduction_indices*
T0*
_output_shapes
:`*
	keep_dims( *

Tidx0
�
Esiamese/scala1/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
3siamese/scala1/moments/sufficient_statistics/var_ssSum>siamese/scala1/moments/sufficient_statistics/SquaredDifferenceEsiamese/scala1/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes
:`*
	keep_dims( *

Tidx0
f
siamese/scala1/moments/ShapeConst*
_output_shapes
:*
valueB:`*
dtype0
�
siamese/scala1/moments/ReshapeReshape#siamese/scala1/moments/StopGradientsiamese/scala1/moments/Shape*
T0*
Tshape0*
_output_shapes
:`
�
(siamese/scala1/moments/normalize/divisor
Reciprocal2siamese/scala1/moments/sufficient_statistics/Const5^siamese/scala1/moments/sufficient_statistics/mean_ss4^siamese/scala1/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
-siamese/scala1/moments/normalize/shifted_meanMul4siamese/scala1/moments/sufficient_statistics/mean_ss(siamese/scala1/moments/normalize/divisor*
T0*
_output_shapes
:`
�
%siamese/scala1/moments/normalize/meanAdd-siamese/scala1/moments/normalize/shifted_meansiamese/scala1/moments/Reshape*
T0*
_output_shapes
:`
�
$siamese/scala1/moments/normalize/MulMul3siamese/scala1/moments/sufficient_statistics/var_ss(siamese/scala1/moments/normalize/divisor*
T0*
_output_shapes
:`
�
'siamese/scala1/moments/normalize/SquareSquare-siamese/scala1/moments/normalize/shifted_mean*
T0*
_output_shapes
:`
�
)siamese/scala1/moments/normalize/varianceSub$siamese/scala1/moments/normalize/Mul'siamese/scala1/moments/normalize/Square*
T0*
_output_shapes
:`
�
$siamese/scala1/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/ConstConst*
valueB`*    *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
:`
�
3siamese/scala1/siamese/scala1/bn/moving_mean/biased
VariableV2*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
:siamese/scala1/siamese/scala1/bn/moving_mean/biased/AssignAssign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Const*
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
Isiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala1/siamese/scala1/bn/moving_mean/local_step
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
	container *
shape: *
dtype0*
_output_shapes
: 
�
>siamese/scala1/siamese/scala1/bn/moving_mean/local_step/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepIsiamese/scala1/siamese/scala1/bn/moving_mean/local_step/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
<siamese/scala1/siamese/scala1/bn/moving_mean/local_step/readIdentity7siamese/scala1/siamese/scala1/bn/moving_mean/local_step*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read%siamese/scala1/moments/normalize/mean*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMul@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub$siamese/scala1/AssignMovingAvg/decay*
_output_shapes
:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
isiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biased@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( 
�
Lsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0
�
Fsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepLsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Asiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x$siamese/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
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
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstj^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanG^siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/x@siamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
Dsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivAsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readDsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
siamese/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanBsiamese/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
_output_shapes
:`*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
&siamese/scala1/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/ConstConst*
valueB`*    *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
:`
�
7siamese/scala1/siamese/scala1/bn/moving_variance/biased
VariableV2*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
	container *
shape:`*
dtype0*
_output_shapes
:`*
shared_name 
�
>siamese/scala1/siamese/scala1/bn/moving_variance/biased/AssignAssign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
<siamese/scala1/siamese/scala1/bn/moving_variance/biased/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biased*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
Msiamese/scala1/siamese/scala1/bn/moving_variance/local_step/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala1/siamese/scala1/bn/moving_variance/local_step
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Bsiamese/scala1/siamese/scala1/bn/moving_variance/local_step/AssignAssign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepMsiamese/scala1/siamese/scala1/bn/moving_variance/local_step/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
@siamese/scala1/siamese/scala1/bn/moving_variance/local_step/readIdentity;siamese/scala1/siamese/scala1/bn/moving_variance/local_step*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
T0
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read)siamese/scala1/moments/normalize/variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub&siamese/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
ssiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedFsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Rsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepRsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Gsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
T0
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
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
Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stept^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Isiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstt^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceM^siamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
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
Jsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivGsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readJsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
 siamese/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceHsiamese/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( *
T0
e
siamese/scala1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

g
siamese/scala1/cond/switch_tIdentitysiamese/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
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
siamese/scala1/cond/Switch_1Switch%siamese/scala1/moments/normalize/meansiamese/scala1/cond/pred_id*
T0*8
_class.
,*loc:@siamese/scala1/moments/normalize/mean* 
_output_shapes
:`:`
�
siamese/scala1/cond/Switch_2Switch)siamese/scala1/moments/normalize/variancesiamese/scala1/cond/pred_id*
T0*<
_class2
0.loc:@siamese/scala1/moments/normalize/variance* 
_output_shapes
:`:`
�
#siamese/scala1/cond/Switch_3/SwitchSwitch"siamese/scala1/bn/moving_mean/readsiamese/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese/scala1/cond/Switch_3Switch#siamese/scala1/cond/Switch_3/Switchsiamese/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
#siamese/scala1/cond/Switch_4/SwitchSwitch&siamese/scala1/bn/moving_variance/readsiamese/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
siamese/scala1/cond/Switch_4Switch#siamese/scala1/cond/Switch_4/Switchsiamese/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
siamese/scala1/cond/MergeMergesiamese/scala1/cond/Switch_3siamese/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese/scala1/cond/Merge_1Mergesiamese/scala1/cond/Switch_4siamese/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
c
siamese/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
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
siamese/scala1/batchnorm/subSubsiamese/scala1/bn/beta/readsiamese/scala1/batchnorm/mul_2*
_output_shapes
:`*
T0
�
siamese/scala1/batchnorm/add_1Addsiamese/scala1/batchnorm/mul_1siamese/scala1/batchnorm/sub*
T0*&
_output_shapes
:;;`
l
siamese/scala1/ReluRelusiamese/scala1/batchnorm/add_1*
T0*&
_output_shapes
:;;`
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
>siamese/scala2/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala2/conv/weights*%
valueB"      0      *
dtype0*
_output_shapes
:
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
dtype0*'
_output_shapes
:0�*

seed *
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
seed2 
�
<siamese/scala2/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala2/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala2/conv/weights/Initializer/truncated_normal/stddev*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�*
T0
�
8siamese/scala2/conv/weights/Initializer/truncated_normalAdd<siamese/scala2/conv/weights/Initializer/truncated_normal/mul=siamese/scala2/conv/weights/Initializer/truncated_normal/mean*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�
�
siamese/scala2/conv/weights
VariableV2*
	container *
shape:0�*
dtype0*'
_output_shapes
:0�*
shared_name *.
_class$
" loc:@siamese/scala2/conv/weights
�
"siamese/scala2/conv/weights/AssignAssignsiamese/scala2/conv/weights8siamese/scala2/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�
�
 siamese/scala2/conv/weights/readIdentitysiamese/scala2/conv/weights*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*'
_output_shapes
:0�
�
<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*.
_class$
" loc:@siamese/scala2/conv/weights*
dtype0*
_output_shapes
: 
�
=siamese/scala2/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala2/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
_output_shapes
: 
�
6siamese/scala2/conv/weights/Regularizer/l2_regularizerMul<siamese/scala2/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala2/conv/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala2/conv/weights
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
VariableV2*
shared_name *-
_class#
!loc:@siamese/scala2/conv/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
!siamese/scala2/conv/biases/AssignAssignsiamese/scala2/conv/biases,siamese/scala2/conv/biases/Initializer/Const*
use_locking(*
T0*-
_class#
!loc:@siamese/scala2/conv/biases*
validate_shape(*
_output_shapes	
:�
�
siamese/scala2/conv/biases/readIdentitysiamese/scala2/conv/biases*
_output_shapes	
:�*
T0*-
_class#
!loc:@siamese/scala2/conv/biases
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
siamese/scala2/Conv2DConv2Dsiamese/scala2/splitsiamese/scala2/split_1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides

�
siamese/scala2/Conv2D_1Conv2Dsiamese/scala2/split:1siamese/scala2/split_1:1*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
\
siamese/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2/concatConcatV2siamese/scala2/Conv2Dsiamese/scala2/Conv2D_1siamese/scala2/concat/axis*
T0*
N*'
_output_shapes
:�*

Tidx0
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
VariableV2*)
_class
loc:@siamese/scala2/bn/beta*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
�
siamese/scala2/bn/beta/AssignAssignsiamese/scala2/bn/beta(siamese/scala2/bn/beta/Initializer/Const*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name **
_class 
loc:@siamese/scala2/bn/gamma*
	container *
shape:�
�
siamese/scala2/bn/gamma/AssignAssignsiamese/scala2/bn/gamma)siamese/scala2/bn/gamma/Initializer/Const**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
siamese/scala2/bn/gamma/readIdentitysiamese/scala2/bn/gamma**
_class 
loc:@siamese/scala2/bn/gamma*
_output_shapes	
:�*
T0
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
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
$siamese/scala2/bn/moving_mean/AssignAssignsiamese/scala2/bn/moving_mean/siamese/scala2/bn/moving_mean/Initializer/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
"siamese/scala2/bn/moving_mean/readIdentitysiamese/scala2/bn/moving_mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
3siamese/scala2/bn/moving_variance/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB�*  �?*
dtype0*
_output_shapes	
:�
�
!siamese/scala2/bn/moving_variance
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
-siamese/scala2/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala2/moments/MeanMeansiamese/scala2/Add-siamese/scala2/moments/Mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
#siamese/scala2/moments/StopGradientStopGradientsiamese/scala2/moments/Mean*
T0*'
_output_shapes
:�
w
2siamese/scala2/moments/sufficient_statistics/ConstConst*
valueB
 * @�E*
dtype0*
_output_shapes
: 
�
0siamese/scala2/moments/sufficient_statistics/SubSubsiamese/scala2/Add#siamese/scala2/moments/StopGradient*'
_output_shapes
:�*
T0
�
>siamese/scala2/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese/scala2/Add#siamese/scala2/moments/StopGradient*
T0*'
_output_shapes
:�
�
Fsiamese/scala2/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
4siamese/scala2/moments/sufficient_statistics/mean_ssSum0siamese/scala2/moments/sufficient_statistics/SubFsiamese/scala2/moments/sufficient_statistics/mean_ss/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
�
Esiamese/scala2/moments/sufficient_statistics/var_ss/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
3siamese/scala2/moments/sufficient_statistics/var_ssSum>siamese/scala2/moments/sufficient_statistics/SquaredDifferenceEsiamese/scala2/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
g
siamese/scala2/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
siamese/scala2/moments/ReshapeReshape#siamese/scala2/moments/StopGradientsiamese/scala2/moments/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
(siamese/scala2/moments/normalize/divisor
Reciprocal2siamese/scala2/moments/sufficient_statistics/Const5^siamese/scala2/moments/sufficient_statistics/mean_ss4^siamese/scala2/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
-siamese/scala2/moments/normalize/shifted_meanMul4siamese/scala2/moments/sufficient_statistics/mean_ss(siamese/scala2/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
%siamese/scala2/moments/normalize/meanAdd-siamese/scala2/moments/normalize/shifted_meansiamese/scala2/moments/Reshape*
T0*
_output_shapes	
:�
�
$siamese/scala2/moments/normalize/MulMul3siamese/scala2/moments/sufficient_statistics/var_ss(siamese/scala2/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
'siamese/scala2/moments/normalize/SquareSquare-siamese/scala2/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
)siamese/scala2/moments/normalize/varianceSub$siamese/scala2/moments/normalize/Mul'siamese/scala2/moments/normalize/Square*
T0*
_output_shapes	
:�
�
$siamese/scala2/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes	
:�
�
3siamese/scala2/siamese/scala2/bn/moving_mean/biased
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
:siamese/scala2/siamese/scala2/bn/moving_mean/biased/AssignAssign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Const*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(
�
8siamese/scala2/siamese/scala2/bn/moving_mean/biased/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Isiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala2/siamese/scala2/bn/moving_mean/local_step
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
>siamese/scala2/siamese/scala2/bn/moving_mean/local_step/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepIsiamese/scala2/siamese/scala2/bn/moving_mean/local_step/Initializer/Const*
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
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read%siamese/scala2/moments/normalize/mean*
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
Lsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0
�
Fsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepLsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Asiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubDsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x$siamese/scala2/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Csiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstj^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanG^siamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
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
siamese/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanBsiamese/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
&siamese/scala2/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
7siamese/scala2/siamese/scala2/bn/moving_variance/biased
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
>siamese/scala2/siamese/scala2/bn/moving_variance/biased/AssignAssign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
<siamese/scala2/siamese/scala2/bn/moving_variance/biased/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biased*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Msiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/ConstConst*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
valueB
 *    *
dtype0*
_output_shapes
: 
�
;siamese/scala2/siamese/scala2/bn/moving_variance/local_step
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Bsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/AssignAssign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepMsiamese/scala2/siamese/scala2/bn/moving_variance/local_step/Initializer/Const*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
@siamese/scala2/siamese/scala2/bn/moving_variance/local_step/readIdentity;siamese/scala2/siamese/scala2/bn/moving_variance/local_step*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read)siamese/scala2/moments/normalize/variance*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mulMulFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub&siamese/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
ssiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
�
Rsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Lsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepRsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Gsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x&siamese/scala2/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stept^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: *
T0
�
Fsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Isiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstt^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceM^siamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xFsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivGsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Hsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readJsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
 siamese/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceHsiamese/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
e
siamese/scala2/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
g
siamese/scala2/cond/switch_tIdentitysiamese/scala2/cond/Switch:1*
T0
*
_output_shapes
: 
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
siamese/scala2/cond/Switch_1Switch%siamese/scala2/moments/normalize/meansiamese/scala2/cond/pred_id*
T0*8
_class.
,*loc:@siamese/scala2/moments/normalize/mean*"
_output_shapes
:�:�
�
siamese/scala2/cond/Switch_2Switch)siamese/scala2/moments/normalize/variancesiamese/scala2/cond/pred_id*<
_class2
0.loc:@siamese/scala2/moments/normalize/variance*"
_output_shapes
:�:�*
T0
�
#siamese/scala2/cond/Switch_3/SwitchSwitch"siamese/scala2/bn/moving_mean/readsiamese/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala2/cond/Switch_3Switch#siamese/scala2/cond/Switch_3/Switchsiamese/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
#siamese/scala2/cond/Switch_4/SwitchSwitch&siamese/scala2/bn/moving_variance/readsiamese/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese/scala2/cond/Switch_4Switch#siamese/scala2/cond/Switch_4/Switchsiamese/scala2/cond/pred_id*
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
T0*
N*
_output_shapes
	:�: 
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
siamese/scala2/batchnorm/mul_2Mulsiamese/scala2/cond/Mergesiamese/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala2/batchnorm/subSubsiamese/scala2/bn/beta/readsiamese/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
siamese/scala2/batchnorm/add_1Addsiamese/scala2/batchnorm/mul_1siamese/scala2/batchnorm/sub*'
_output_shapes
:�*
T0
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
>siamese/scala3/conv/weights/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*.
_class$
" loc:@siamese/scala3/conv/weights*%
valueB"         �  
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
?siamese/scala3/conv/weights/Initializer/truncated_normal/stddevConst*.
_class$
" loc:@siamese/scala3/conv/weights*
valueB
 *���<*
dtype0*
_output_shapes
: 
�
Hsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala3/conv/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:��*

seed *
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
seed2 
�
<siamese/scala3/conv/weights/Initializer/truncated_normal/mulMulHsiamese/scala3/conv/weights/Initializer/truncated_normal/TruncatedNormal?siamese/scala3/conv/weights/Initializer/truncated_normal/stddev*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*(
_output_shapes
:��
�
8siamese/scala3/conv/weights/Initializer/truncated_normalAdd<siamese/scala3/conv/weights/Initializer/truncated_normal/mul=siamese/scala3/conv/weights/Initializer/truncated_normal/mean*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
siamese/scala3/conv/weights
VariableV2*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala3/conv/weights*
	container *
shape:��
�
"siamese/scala3/conv/weights/AssignAssignsiamese/scala3/conv/weights8siamese/scala3/conv/weights/Initializer/truncated_normal*
use_locking(*
T0*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��
�
 siamese/scala3/conv/weights/readIdentitysiamese/scala3/conv/weights*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala3/conv/weights
�
<siamese/scala3/conv/weights/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*.
_class$
" loc:@siamese/scala3/conv/weights*
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
,siamese/scala3/conv/biases/Initializer/ConstConst*
_output_shapes	
:�*-
_class#
!loc:@siamese/scala3/conv/biases*
valueB�*���=*
dtype0
�
siamese/scala3/conv/biases
VariableV2*-
_class#
!loc:@siamese/scala3/conv/biases*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name 
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
siamese/scala3/Conv2DConv2Dsiamese/scala2/poll/MaxPool siamese/scala3/conv/weights/read*'
_output_shapes
:

�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala3/AddAddsiamese/scala3/Conv2Dsiamese/scala3/conv/biases/read*'
_output_shapes
:

�*
T0
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala3/bn/beta*
	container *
shape:�
�
siamese/scala3/bn/beta/AssignAssignsiamese/scala3/bn/beta(siamese/scala3/bn/beta/Initializer/Const*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(*
_output_shapes	
:�
�
siamese/scala3/bn/beta/readIdentitysiamese/scala3/bn/beta*
T0*)
_class
loc:@siamese/scala3/bn/beta*
_output_shapes	
:�
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name **
_class 
loc:@siamese/scala3/bn/gamma*
	container *
shape:�
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
/siamese/scala3/bn/moving_mean/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
valueB�*    *
dtype0*
_output_shapes	
:�
�
siamese/scala3/bn/moving_mean
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
$siamese/scala3/bn/moving_mean/AssignAssignsiamese/scala3/bn/moving_mean/siamese/scala3/bn/moving_mean/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape:�
�
(siamese/scala3/bn/moving_variance/AssignAssign!siamese/scala3/bn/moving_variance3siamese/scala3/bn/moving_variance/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
&siamese/scala3/bn/moving_variance/readIdentity!siamese/scala3/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala3/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala3/moments/MeanMeansiamese/scala3/Add-siamese/scala3/moments/Mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
#siamese/scala3/moments/StopGradientStopGradientsiamese/scala3/moments/Mean*'
_output_shapes
:�*
T0
w
2siamese/scala3/moments/sufficient_statistics/ConstConst*
valueB
 *  HD*
dtype0*
_output_shapes
: 
�
0siamese/scala3/moments/sufficient_statistics/SubSubsiamese/scala3/Add#siamese/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
�
>siamese/scala3/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese/scala3/Add#siamese/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
�
Fsiamese/scala3/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
4siamese/scala3/moments/sufficient_statistics/mean_ssSum0siamese/scala3/moments/sufficient_statistics/SubFsiamese/scala3/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
Esiamese/scala3/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
3siamese/scala3/moments/sufficient_statistics/var_ssSum>siamese/scala3/moments/sufficient_statistics/SquaredDifferenceEsiamese/scala3/moments/sufficient_statistics/var_ss/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
g
siamese/scala3/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
siamese/scala3/moments/ReshapeReshape#siamese/scala3/moments/StopGradientsiamese/scala3/moments/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
(siamese/scala3/moments/normalize/divisor
Reciprocal2siamese/scala3/moments/sufficient_statistics/Const5^siamese/scala3/moments/sufficient_statistics/mean_ss4^siamese/scala3/moments/sufficient_statistics/var_ss*
_output_shapes
: *
T0
�
-siamese/scala3/moments/normalize/shifted_meanMul4siamese/scala3/moments/sufficient_statistics/mean_ss(siamese/scala3/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
%siamese/scala3/moments/normalize/meanAdd-siamese/scala3/moments/normalize/shifted_meansiamese/scala3/moments/Reshape*
_output_shapes	
:�*
T0
�
$siamese/scala3/moments/normalize/MulMul3siamese/scala3/moments/sufficient_statistics/var_ss(siamese/scala3/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
'siamese/scala3/moments/normalize/SquareSquare-siamese/scala3/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
)siamese/scala3/moments/normalize/varianceSub$siamese/scala3/moments/normalize/Mul'siamese/scala3/moments/normalize/Square*
_output_shapes	
:�*
T0
�
$siamese/scala3/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes	
:�
�
3siamese/scala3/siamese/scala3/bn/moving_mean/biased
VariableV2*
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape:�*
dtype0*
_output_shapes	
:�
�
:siamese/scala3/siamese/scala3/bn/moving_mean/biased/AssignAssign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Const*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
8siamese/scala3/siamese/scala3/bn/moving_mean/biased/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biased*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Isiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/ConstConst*0
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
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
	container *
shape: 
�
>siamese/scala3/siamese/scala3/bn/moving_mean/local_step/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepIsiamese/scala3/siamese/scala3/bn/moving_mean/local_step/Initializer/Const*
_output_shapes
: *
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(
�
<siamese/scala3/siamese/scala3/bn/moving_mean/local_step/readIdentity7siamese/scala3/siamese/scala3/bn/moving_mean/local_step*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read%siamese/scala3/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMul@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub$siamese/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
isiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biased@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Lsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Fsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepLsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Asiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Bsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubDsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x$siamese/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Csiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstj^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanG^siamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0
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
siamese/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanBsiamese/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
&siamese/scala3/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
7siamese/scala3/siamese/scala3/bn/moving_variance/biased
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
	container *
shape:�
�
>siamese/scala3/siamese/scala3/bn/moving_variance/biased/AssignAssign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Const*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(
�
<siamese/scala3/siamese/scala3/bn/moving_variance/biased/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biased*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
Msiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/ConstConst*4
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
Bsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/AssignAssign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepMsiamese/scala3/siamese/scala3/bn/moving_variance/local_step/Initializer/Const*
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
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read)siamese/scala3/moments/normalize/variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Fsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub&siamese/scala3/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
ssiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Rsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
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
Gsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
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
Isiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stept^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
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
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstt^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceM^siamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubJsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xFsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Jsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivGsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readHsiamese/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
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
siamese/scala3/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
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
siamese/scala3/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala3/cond/Switch_1Switch%siamese/scala3/moments/normalize/meansiamese/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*8
_class.
,*loc:@siamese/scala3/moments/normalize/mean
�
siamese/scala3/cond/Switch_2Switch)siamese/scala3/moments/normalize/variancesiamese/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*<
_class2
0.loc:@siamese/scala3/moments/normalize/variance
�
#siamese/scala3/cond/Switch_3/SwitchSwitch"siamese/scala3/bn/moving_mean/readsiamese/scala3/cond/pred_id*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese/scala3/cond/Switch_3Switch#siamese/scala3/cond/Switch_3/Switchsiamese/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
#siamese/scala3/cond/Switch_4/SwitchSwitch&siamese/scala3/bn/moving_variance/readsiamese/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala3/cond/Switch_4Switch#siamese/scala3/cond/Switch_4/Switchsiamese/scala3/cond/pred_id*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese/scala3/cond/MergeMergesiamese/scala3/cond/Switch_3siamese/scala3/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala3/cond/Merge_1Mergesiamese/scala3/cond/Switch_4siamese/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
c
siamese/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala3/batchnorm/addAddsiamese/scala3/cond/Merge_1siamese/scala3/batchnorm/add/y*
T0*
_output_shapes	
:�
k
siamese/scala3/batchnorm/RsqrtRsqrtsiamese/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/mulMulsiamese/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
siamese/scala3/batchnorm/mul_1Mulsiamese/scala3/Addsiamese/scala3/batchnorm/mul*'
_output_shapes
:

�*
T0
�
siamese/scala3/batchnorm/mul_2Mulsiamese/scala3/cond/Mergesiamese/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/subSubsiamese/scala3/bn/beta/readsiamese/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
siamese/scala3/batchnorm/add_1Addsiamese/scala3/batchnorm/mul_1siamese/scala3/batchnorm/sub*'
_output_shapes
:

�*
T0
m
siamese/scala3/ReluRelusiamese/scala3/batchnorm/add_1*'
_output_shapes
:

�*
T0
�
>siamese/scala4/conv/weights/Initializer/truncated_normal/shapeConst*.
_class$
" loc:@siamese/scala4/conv/weights*%
valueB"      �   �  *
dtype0*
_output_shapes
:
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
Hsiamese/scala4/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala4/conv/weights/Initializer/truncated_normal/shape*
seed2 *
dtype0*(
_output_shapes
:��*

seed *
T0*.
_class$
" loc:@siamese/scala4/conv/weights
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
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala4/conv/weights*
	container *
shape:��
�
"siamese/scala4/conv/weights/AssignAssignsiamese/scala4/conv/weights8siamese/scala4/conv/weights/Initializer/truncated_normal*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0*.
_class$
" loc:@siamese/scala4/conv/weights
�
 siamese/scala4/conv/weights/readIdentitysiamese/scala4/conv/weights*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*(
_output_shapes
:��
�
<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*.
_class$
" loc:@siamese/scala4/conv/weights*
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
6siamese/scala4/conv/weights/Regularizer/l2_regularizerMul<siamese/scala4/conv/weights/Regularizer/l2_regularizer/scale=siamese/scala4/conv/weights/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*.
_class$
" loc:@siamese/scala4/conv/weights
�
,siamese/scala4/conv/biases/Initializer/ConstConst*-
_class#
!loc:@siamese/scala4/conv/biases*
valueB�*���=*
dtype0*
_output_shapes	
:�
�
siamese/scala4/conv/biases
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala4/conv/biases*
	container *
shape:�
�
!siamese/scala4/conv/biases/AssignAssignsiamese/scala4/conv/biases,siamese/scala4/conv/biases/Initializer/Const*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(
�
siamese/scala4/conv/biases/readIdentitysiamese/scala4/conv/biases*
_output_shapes	
:�*
T0*-
_class#
!loc:@siamese/scala4/conv/biases
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
b
 siamese/scala4/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/split_1Split siamese/scala4/split_1/split_dim siamese/scala4/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese/scala4/Conv2DConv2Dsiamese/scala4/splitsiamese/scala4/split_1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala4/Conv2D_1Conv2Dsiamese/scala4/split:1siamese/scala4/split_1:1*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
\
siamese/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4/concatConcatV2siamese/scala4/Conv2Dsiamese/scala4/Conv2D_1siamese/scala4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese/scala4/AddAddsiamese/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
(siamese/scala4/bn/beta/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*)
_class
loc:@siamese/scala4/bn/beta*
valueB�*    
�
siamese/scala4/bn/beta
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *)
_class
loc:@siamese/scala4/bn/beta*
	container *
shape:�
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
)siamese/scala4/bn/gamma/Initializer/ConstConst*
dtype0*
_output_shapes	
:�**
_class 
loc:@siamese/scala4/bn/gamma*
valueB�*  �?
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
siamese/scala4/bn/gamma/AssignAssignsiamese/scala4/bn/gamma)siamese/scala4/bn/gamma/Initializer/Const*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma
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
VariableV2*
_output_shapes	
:�*
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape:�*
dtype0
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
3siamese/scala4/bn/moving_variance/Initializer/ConstConst*
dtype0*
_output_shapes	
:�*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB�*  �?
�
!siamese/scala4/bn/moving_variance
VariableV2*
	container *
shape:�*
dtype0*
_output_shapes	
:�*
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
(siamese/scala4/bn/moving_variance/AssignAssign!siamese/scala4/bn/moving_variance3siamese/scala4/bn/moving_variance/Initializer/Const*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(
�
&siamese/scala4/bn/moving_variance/readIdentity!siamese/scala4/bn/moving_variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
-siamese/scala4/moments/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese/scala4/moments/MeanMeansiamese/scala4/Add-siamese/scala4/moments/Mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
#siamese/scala4/moments/StopGradientStopGradientsiamese/scala4/moments/Mean*
T0*'
_output_shapes
:�
w
2siamese/scala4/moments/sufficient_statistics/ConstConst*
_output_shapes
: *
valueB
 *   D*
dtype0
�
0siamese/scala4/moments/sufficient_statistics/SubSubsiamese/scala4/Add#siamese/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
>siamese/scala4/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese/scala4/Add#siamese/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
Fsiamese/scala4/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
4siamese/scala4/moments/sufficient_statistics/mean_ssSum0siamese/scala4/moments/sufficient_statistics/SubFsiamese/scala4/moments/sufficient_statistics/mean_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
Esiamese/scala4/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
3siamese/scala4/moments/sufficient_statistics/var_ssSum>siamese/scala4/moments/sufficient_statistics/SquaredDifferenceEsiamese/scala4/moments/sufficient_statistics/var_ss/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes	
:�
g
siamese/scala4/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
siamese/scala4/moments/ReshapeReshape#siamese/scala4/moments/StopGradientsiamese/scala4/moments/Shape*
_output_shapes	
:�*
T0*
Tshape0
�
(siamese/scala4/moments/normalize/divisor
Reciprocal2siamese/scala4/moments/sufficient_statistics/Const5^siamese/scala4/moments/sufficient_statistics/mean_ss4^siamese/scala4/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
-siamese/scala4/moments/normalize/shifted_meanMul4siamese/scala4/moments/sufficient_statistics/mean_ss(siamese/scala4/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
%siamese/scala4/moments/normalize/meanAdd-siamese/scala4/moments/normalize/shifted_meansiamese/scala4/moments/Reshape*
_output_shapes	
:�*
T0
�
$siamese/scala4/moments/normalize/MulMul3siamese/scala4/moments/sufficient_statistics/var_ss(siamese/scala4/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
'siamese/scala4/moments/normalize/SquareSquare-siamese/scala4/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
)siamese/scala4/moments/normalize/varianceSub$siamese/scala4/moments/normalize/Mul'siamese/scala4/moments/normalize/Square*
T0*
_output_shapes	
:�
�
$siamese/scala4/AssignMovingAvg/decayConst*
_output_shapes
: *
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
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
:siamese/scala4/siamese/scala4/bn/moving_mean/biased/AssignAssign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Const*
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
Isiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/ConstConst*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
valueB
 *    *
dtype0*
_output_shapes
: 
�
7siamese/scala4/siamese/scala4/bn/moving_mean/local_step
VariableV2*
dtype0*
_output_shapes
: *
shared_name *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
	container *
shape: 
�
>siamese/scala4/siamese/scala4/bn/moving_mean/local_step/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepIsiamese/scala4/siamese/scala4/bn/moving_mean/local_step/Initializer/Const*
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
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read%siamese/scala4/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMul@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub$siamese/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
isiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biased@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Lsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Fsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepLsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Asiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x$siamese/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Csiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstj^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanG^siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/x@siamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Dsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivAsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readDsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
siamese/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanBsiamese/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
&siamese/scala4/AssignMovingAvg_1/decayConst*
_output_shapes
: *
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/ConstConst*
_output_shapes	
:�*
valueB�*    *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0
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
>siamese/scala4/siamese/scala4/bn/moving_variance/biased/AssignAssign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Const*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
<siamese/scala4/siamese/scala4/bn/moving_variance/biased/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biased*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Msiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/ConstConst*
_output_shapes
: *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
valueB
 *    *
dtype0
�
;siamese/scala4/siamese/scala4/bn/moving_variance/local_step
VariableV2*
shared_name *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
	container *
shape: *
dtype0*
_output_shapes
: 
�
Bsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/AssignAssign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepMsiamese/scala4/siamese/scala4/bn/moving_variance/local_step/Initializer/Const*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
@siamese/scala4/siamese/scala4/bn/moving_variance/local_step/readIdentity;siamese/scala4/siamese/scala4/bn/moving_variance/local_step*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read)siamese/scala4/moments/normalize/variance*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub&siamese/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
ssiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedFsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Rsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0
�
Lsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepRsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Gsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x&siamese/scala4/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stept^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Fsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Isiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstt^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceM^siamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
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
Jsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivGsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readJsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
 siamese/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceHsiamese/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
e
siamese/scala4/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

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
siamese/scala4/cond/Switch_1Switch%siamese/scala4/moments/normalize/meansiamese/scala4/cond/pred_id*
T0*8
_class.
,*loc:@siamese/scala4/moments/normalize/mean*"
_output_shapes
:�:�
�
siamese/scala4/cond/Switch_2Switch)siamese/scala4/moments/normalize/variancesiamese/scala4/cond/pred_id*
T0*<
_class2
0.loc:@siamese/scala4/moments/normalize/variance*"
_output_shapes
:�:�
�
#siamese/scala4/cond/Switch_3/SwitchSwitch"siamese/scala4/bn/moving_mean/readsiamese/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
siamese/scala4/cond/Switch_3Switch#siamese/scala4/cond/Switch_3/Switchsiamese/scala4/cond/pred_id*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
#siamese/scala4/cond/Switch_4/SwitchSwitch&siamese/scala4/bn/moving_variance/readsiamese/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese/scala4/cond/Switch_4Switch#siamese/scala4/cond/Switch_4/Switchsiamese/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala4/cond/MergeMergesiamese/scala4/cond/Switch_3siamese/scala4/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala4/cond/Merge_1Mergesiamese/scala4/cond/Switch_4siamese/scala4/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
c
siamese/scala4/batchnorm/add/yConst*
_output_shapes
: *
valueB
 *o�:*
dtype0
�
siamese/scala4/batchnorm/addAddsiamese/scala4/cond/Merge_1siamese/scala4/batchnorm/add/y*
T0*
_output_shapes	
:�
k
siamese/scala4/batchnorm/RsqrtRsqrtsiamese/scala4/batchnorm/add*
_output_shapes	
:�*
T0
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
siamese/scala4/batchnorm/mul_2Mulsiamese/scala4/cond/Mergesiamese/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala4/batchnorm/subSubsiamese/scala4/bn/beta/readsiamese/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
siamese/scala4/batchnorm/add_1Addsiamese/scala4/batchnorm/mul_1siamese/scala4/batchnorm/sub*'
_output_shapes
:�*
T0
m
siamese/scala4/ReluRelusiamese/scala4/batchnorm/add_1*'
_output_shapes
:�*
T0
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
Hsiamese/scala5/conv/weights/Initializer/truncated_normal/TruncatedNormalTruncatedNormal>siamese/scala5/conv/weights/Initializer/truncated_normal/shape*
dtype0*(
_output_shapes
:��*

seed *
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
seed2 
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
VariableV2*
dtype0*(
_output_shapes
:��*
shared_name *.
_class$
" loc:@siamese/scala5/conv/weights*
	container *
shape:��
�
"siamese/scala5/conv/weights/AssignAssignsiamese/scala5/conv/weights8siamese/scala5/conv/weights/Initializer/truncated_normal*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
�
 siamese/scala5/conv/weights/readIdentitysiamese/scala5/conv/weights*(
_output_shapes
:��*
T0*.
_class$
" loc:@siamese/scala5/conv/weights
�
<siamese/scala5/conv/weights/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*.
_class$
" loc:@siamese/scala5/conv/weights*
dtype0*
_output_shapes
: 
�
=siamese/scala5/conv/weights/Regularizer/l2_regularizer/L2LossL2Loss siamese/scala5/conv/weights/read*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
_output_shapes
: 
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
VariableV2*
dtype0*
_output_shapes	
:�*
shared_name *-
_class#
!loc:@siamese/scala5/conv/biases*
	container *
shape:�
�
!siamese/scala5/conv/biases/AssignAssignsiamese/scala5/conv/biases,siamese/scala5/conv/biases/Initializer/Const*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
siamese/scala5/conv/biases/readIdentitysiamese/scala5/conv/biases*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
_output_shapes	
:�
`
siamese/scala5/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese/scala5/splitSplitsiamese/scala5/split/split_dimsiamese/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
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
siamese/scala5/Conv2DConv2Dsiamese/scala5/splitsiamese/scala5/split_1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala5/Conv2D_1Conv2Dsiamese/scala5/split:1siamese/scala5/split_1:1*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
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
siamese/scala1_1/Conv2DConv2DPlaceholder_3 siamese/scala1/conv/weights/read*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:{{`
�
siamese/scala1_1/AddAddsiamese/scala1_1/Conv2Dsiamese/scala1/conv/biases/read*&
_output_shapes
:{{`*
T0
�
/siamese/scala1_1/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala1_1/moments/MeanMeansiamese/scala1_1/Add/siamese/scala1_1/moments/Mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
%siamese/scala1_1/moments/StopGradientStopGradientsiamese/scala1_1/moments/Mean*
T0*&
_output_shapes
:`
y
4siamese/scala1_1/moments/sufficient_statistics/ConstConst*
valueB
 * d�G*
dtype0*
_output_shapes
: 
�
2siamese/scala1_1/moments/sufficient_statistics/SubSubsiamese/scala1_1/Add%siamese/scala1_1/moments/StopGradient*
T0*&
_output_shapes
:{{`
�
@siamese/scala1_1/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese/scala1_1/Add%siamese/scala1_1/moments/StopGradient*&
_output_shapes
:{{`*
T0
�
Hsiamese/scala1_1/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese/scala1_1/moments/sufficient_statistics/mean_ssSum2siamese/scala1_1/moments/sufficient_statistics/SubHsiamese/scala1_1/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes
:`*
	keep_dims( *

Tidx0*
T0
�
Gsiamese/scala1_1/moments/sufficient_statistics/var_ss/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
5siamese/scala1_1/moments/sufficient_statistics/var_ssSum@siamese/scala1_1/moments/sufficient_statistics/SquaredDifferenceGsiamese/scala1_1/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes
:`*
	keep_dims( *

Tidx0
h
siamese/scala1_1/moments/ShapeConst*
valueB:`*
dtype0*
_output_shapes
:
�
 siamese/scala1_1/moments/ReshapeReshape%siamese/scala1_1/moments/StopGradientsiamese/scala1_1/moments/Shape*
T0*
Tshape0*
_output_shapes
:`
�
*siamese/scala1_1/moments/normalize/divisor
Reciprocal4siamese/scala1_1/moments/sufficient_statistics/Const7^siamese/scala1_1/moments/sufficient_statistics/mean_ss6^siamese/scala1_1/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
/siamese/scala1_1/moments/normalize/shifted_meanMul6siamese/scala1_1/moments/sufficient_statistics/mean_ss*siamese/scala1_1/moments/normalize/divisor*
T0*
_output_shapes
:`
�
'siamese/scala1_1/moments/normalize/meanAdd/siamese/scala1_1/moments/normalize/shifted_mean siamese/scala1_1/moments/Reshape*
T0*
_output_shapes
:`
�
&siamese/scala1_1/moments/normalize/MulMul5siamese/scala1_1/moments/sufficient_statistics/var_ss*siamese/scala1_1/moments/normalize/divisor*
T0*
_output_shapes
:`
�
)siamese/scala1_1/moments/normalize/SquareSquare/siamese/scala1_1/moments/normalize/shifted_mean*
_output_shapes
:`*
T0
�
+siamese/scala1_1/moments/normalize/varianceSub&siamese/scala1_1/moments/normalize/Mul)siamese/scala1_1/moments/normalize/Square*
_output_shapes
:`*
T0
�
&siamese/scala1_1/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/ConstConst*
valueB`*    *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
:`
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read'siamese/scala1_1/moments/normalize/mean*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese/scala1_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Nsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Hsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
use_locking( 
�
Csiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
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
Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstl^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese/scala1_1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
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
(siamese/scala1_1/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/ConstConst*
_output_shapes
:`*
valueB`*    *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read+siamese/scala1_1/moments/normalize/variance*
_output_shapes
:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese/scala1_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
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
Isiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
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
Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstv^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese/scala1_1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese/scala1_1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
g
siamese/scala1_1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
k
siamese/scala1_1/cond/switch_tIdentitysiamese/scala1_1/cond/Switch:1*
T0
*
_output_shapes
: 
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
siamese/scala1_1/cond/Switch_1Switch'siamese/scala1_1/moments/normalize/meansiamese/scala1_1/cond/pred_id*
T0*:
_class0
.,loc:@siamese/scala1_1/moments/normalize/mean* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_2Switch+siamese/scala1_1/moments/normalize/variancesiamese/scala1_1/cond/pred_id*>
_class4
20loc:@siamese/scala1_1/moments/normalize/variance* 
_output_shapes
:`:`*
T0
�
%siamese/scala1_1/cond/Switch_3/SwitchSwitch"siamese/scala1/bn/moving_mean/readsiamese/scala1_1/cond/pred_id* 
_output_shapes
:`:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
siamese/scala1_1/cond/Switch_3Switch%siamese/scala1_1/cond/Switch_3/Switchsiamese/scala1_1/cond/pred_id*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`*
T0
�
%siamese/scala1_1/cond/Switch_4/SwitchSwitch&siamese/scala1/bn/moving_variance/readsiamese/scala1_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese/scala1_1/cond/Switch_4Switch%siamese/scala1_1/cond/Switch_4/Switchsiamese/scala1_1/cond/pred_id*
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
siamese/scala1_1/batchnorm/addAddsiamese/scala1_1/cond/Merge_1 siamese/scala1_1/batchnorm/add/y*
T0*
_output_shapes
:`
n
 siamese/scala1_1/batchnorm/RsqrtRsqrtsiamese/scala1_1/batchnorm/add*
_output_shapes
:`*
T0
�
siamese/scala1_1/batchnorm/mulMul siamese/scala1_1/batchnorm/Rsqrtsiamese/scala1/bn/gamma/read*
_output_shapes
:`*
T0
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
siamese/scala1_1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese/scala1_1/batchnorm/mul_2*
_output_shapes
:`*
T0
�
 siamese/scala1_1/batchnorm/add_1Add siamese/scala1_1/batchnorm/mul_1siamese/scala1_1/batchnorm/sub*
T0*&
_output_shapes
:{{`
p
siamese/scala1_1/ReluRelu siamese/scala1_1/batchnorm/add_1*
T0*&
_output_shapes
:{{`
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
d
"siamese/scala2_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala2_1/split_1Split"siamese/scala2_1/split_1/split_dim siamese/scala2/conv/weights/read*
T0*:
_output_shapes(
&:0�:0�*
	num_split
�
siamese/scala2_1/Conv2DConv2Dsiamese/scala2_1/splitsiamese/scala2_1/split_1*
paddingVALID*'
_output_shapes
:99�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese/scala2_1/Conv2D_1Conv2Dsiamese/scala2_1/split:1siamese/scala2_1/split_1:1*
paddingVALID*'
_output_shapes
:99�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
^
siamese/scala2_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
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
/siamese/scala2_1/moments/Mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese/scala2_1/moments/MeanMeansiamese/scala2_1/Add/siamese/scala2_1/moments/Mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese/scala2_1/moments/StopGradientStopGradientsiamese/scala2_1/moments/Mean*
T0*'
_output_shapes
:�
y
4siamese/scala2_1/moments/sufficient_statistics/ConstConst*
_output_shapes
: *
valueB
 * �F*
dtype0
�
2siamese/scala2_1/moments/sufficient_statistics/SubSubsiamese/scala2_1/Add%siamese/scala2_1/moments/StopGradient*
T0*'
_output_shapes
:99�
�
@siamese/scala2_1/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese/scala2_1/Add%siamese/scala2_1/moments/StopGradient*
T0*'
_output_shapes
:99�
�
Hsiamese/scala2_1/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese/scala2_1/moments/sufficient_statistics/mean_ssSum2siamese/scala2_1/moments/sufficient_statistics/SubHsiamese/scala2_1/moments/sufficient_statistics/mean_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
Gsiamese/scala2_1/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese/scala2_1/moments/sufficient_statistics/var_ssSum@siamese/scala2_1/moments/sufficient_statistics/SquaredDifferenceGsiamese/scala2_1/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
i
siamese/scala2_1/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
 siamese/scala2_1/moments/ReshapeReshape%siamese/scala2_1/moments/StopGradientsiamese/scala2_1/moments/Shape*
Tshape0*
_output_shapes	
:�*
T0
�
*siamese/scala2_1/moments/normalize/divisor
Reciprocal4siamese/scala2_1/moments/sufficient_statistics/Const7^siamese/scala2_1/moments/sufficient_statistics/mean_ss6^siamese/scala2_1/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
/siamese/scala2_1/moments/normalize/shifted_meanMul6siamese/scala2_1/moments/sufficient_statistics/mean_ss*siamese/scala2_1/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
'siamese/scala2_1/moments/normalize/meanAdd/siamese/scala2_1/moments/normalize/shifted_mean siamese/scala2_1/moments/Reshape*
T0*
_output_shapes	
:�
�
&siamese/scala2_1/moments/normalize/MulMul5siamese/scala2_1/moments/sufficient_statistics/var_ss*siamese/scala2_1/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
)siamese/scala2_1/moments/normalize/SquareSquare/siamese/scala2_1/moments/normalize/shifted_mean*
_output_shapes	
:�*
T0
�
+siamese/scala2_1/moments/normalize/varianceSub&siamese/scala2_1/moments/normalize/Mul)siamese/scala2_1/moments/normalize/Square*
_output_shapes	
:�*
T0
�
&siamese/scala2_1/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    *0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read'siamese/scala2_1/moments/normalize/mean*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese/scala2_1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
ksiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Nsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
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
Csiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
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
Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: *
T0
�
Bsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstl^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
Dsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
T0
�
 siamese/scala2_1/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese/scala2_1/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
(siamese/scala2_1/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read+siamese/scala2_1/moments/normalize/variance*
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
usiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Nsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepTsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Isiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
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
Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstv^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
"siamese/scala2_1/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese/scala2_1/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
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
siamese/scala2_1/cond/switch_fIdentitysiamese/scala2_1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese/scala2_1/cond/pred_idIdentityis_training_1*
T0
*
_output_shapes
: 
�
siamese/scala2_1/cond/Switch_1Switch'siamese/scala2_1/moments/normalize/meansiamese/scala2_1/cond/pred_id*"
_output_shapes
:�:�*
T0*:
_class0
.,loc:@siamese/scala2_1/moments/normalize/mean
�
siamese/scala2_1/cond/Switch_2Switch+siamese/scala2_1/moments/normalize/variancesiamese/scala2_1/cond/pred_id*
T0*>
_class4
20loc:@siamese/scala2_1/moments/normalize/variance*"
_output_shapes
:�:�
�
%siamese/scala2_1/cond/Switch_3/SwitchSwitch"siamese/scala2/bn/moving_mean/readsiamese/scala2_1/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
siamese/scala2_1/cond/Switch_3Switch%siamese/scala2_1/cond/Switch_3/Switchsiamese/scala2_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
%siamese/scala2_1/cond/Switch_4/SwitchSwitch&siamese/scala2/bn/moving_variance/readsiamese/scala2_1/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese/scala2_1/cond/Switch_4Switch%siamese/scala2_1/cond/Switch_4/Switchsiamese/scala2_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala2_1/cond/MergeMergesiamese/scala2_1/cond/Switch_3 siamese/scala2_1/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese/scala2_1/cond/Merge_1Mergesiamese/scala2_1/cond/Switch_4 siamese/scala2_1/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese/scala2_1/batchnorm/add/yConst*
dtype0*
_output_shapes
: *
valueB
 *o�:
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
 siamese/scala2_1/batchnorm/mul_1Mulsiamese/scala2_1/Addsiamese/scala2_1/batchnorm/mul*'
_output_shapes
:99�*
T0
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
 siamese/scala2_1/batchnorm/add_1Add siamese/scala2_1/batchnorm/mul_1siamese/scala2_1/batchnorm/sub*'
_output_shapes
:99�*
T0
q
siamese/scala2_1/ReluRelu siamese/scala2_1/batchnorm/add_1*
T0*'
_output_shapes
:99�
�
siamese/scala2_1/poll/MaxPoolMaxPoolsiamese/scala2_1/Relu*
ksize
*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides

�
siamese/scala3_1/Conv2DConv2Dsiamese/scala2_1/poll/MaxPool siamese/scala3/conv/weights/read*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
T0
�
siamese/scala3_1/AddAddsiamese/scala3_1/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese/scala3_1/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese/scala3_1/moments/MeanMeansiamese/scala3_1/Add/siamese/scala3_1/moments/Mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese/scala3_1/moments/StopGradientStopGradientsiamese/scala3_1/moments/Mean*
T0*'
_output_shapes
:�
y
4siamese/scala3_1/moments/sufficient_statistics/ConstConst*
valueB
 *  �E*
dtype0*
_output_shapes
: 
�
2siamese/scala3_1/moments/sufficient_statistics/SubSubsiamese/scala3_1/Add%siamese/scala3_1/moments/StopGradient*
T0*'
_output_shapes
:�
�
@siamese/scala3_1/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese/scala3_1/Add%siamese/scala3_1/moments/StopGradient*
T0*'
_output_shapes
:�
�
Hsiamese/scala3_1/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese/scala3_1/moments/sufficient_statistics/mean_ssSum2siamese/scala3_1/moments/sufficient_statistics/SubHsiamese/scala3_1/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
Gsiamese/scala3_1/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese/scala3_1/moments/sufficient_statistics/var_ssSum@siamese/scala3_1/moments/sufficient_statistics/SquaredDifferenceGsiamese/scala3_1/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
i
siamese/scala3_1/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
 siamese/scala3_1/moments/ReshapeReshape%siamese/scala3_1/moments/StopGradientsiamese/scala3_1/moments/Shape*
_output_shapes	
:�*
T0*
Tshape0
�
*siamese/scala3_1/moments/normalize/divisor
Reciprocal4siamese/scala3_1/moments/sufficient_statistics/Const7^siamese/scala3_1/moments/sufficient_statistics/mean_ss6^siamese/scala3_1/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
/siamese/scala3_1/moments/normalize/shifted_meanMul6siamese/scala3_1/moments/sufficient_statistics/mean_ss*siamese/scala3_1/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
'siamese/scala3_1/moments/normalize/meanAdd/siamese/scala3_1/moments/normalize/shifted_mean siamese/scala3_1/moments/Reshape*
_output_shapes	
:�*
T0
�
&siamese/scala3_1/moments/normalize/MulMul5siamese/scala3_1/moments/sufficient_statistics/var_ss*siamese/scala3_1/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
)siamese/scala3_1/moments/normalize/SquareSquare/siamese/scala3_1/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
+siamese/scala3_1/moments/normalize/varianceSub&siamese/scala3_1/moments/normalize/Mul)siamese/scala3_1/moments/normalize/Square*
T0*
_output_shapes	
:�
�
&siamese/scala3_1/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    *0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read'siamese/scala3_1/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese/scala3_1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Nsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Hsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Csiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
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
Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstl^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese/scala3_1/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese/scala3_1/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
(siamese/scala3_1/AssignMovingAvg_1/decayConst*
dtype0*
_output_shapes
: *
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/ConstConst*
_output_shapes	
:�*
valueB�*    *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read+siamese/scala3_1/moments/normalize/variance*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese/scala3_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Nsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese/scala3_1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Hsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstv^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Jsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Lsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese/scala3_1/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
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
siamese/scala3_1/cond/SwitchSwitchis_training_1is_training_1*
_output_shapes
: : *
T0

k
siamese/scala3_1/cond/switch_tIdentitysiamese/scala3_1/cond/Switch:1*
T0
*
_output_shapes
: 
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
siamese/scala3_1/cond/Switch_1Switch'siamese/scala3_1/moments/normalize/meansiamese/scala3_1/cond/pred_id*
T0*:
_class0
.,loc:@siamese/scala3_1/moments/normalize/mean*"
_output_shapes
:�:�
�
siamese/scala3_1/cond/Switch_2Switch+siamese/scala3_1/moments/normalize/variancesiamese/scala3_1/cond/pred_id*"
_output_shapes
:�:�*
T0*>
_class4
20loc:@siamese/scala3_1/moments/normalize/variance
�
%siamese/scala3_1/cond/Switch_3/SwitchSwitch"siamese/scala3/bn/moving_mean/readsiamese/scala3_1/cond/pred_id*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
siamese/scala3_1/cond/Switch_3Switch%siamese/scala3_1/cond/Switch_3/Switchsiamese/scala3_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
%siamese/scala3_1/cond/Switch_4/SwitchSwitch&siamese/scala3/bn/moving_variance/readsiamese/scala3_1/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese/scala3_1/cond/Switch_4Switch%siamese/scala3_1/cond/Switch_4/Switchsiamese/scala3_1/cond/pred_id*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�*
T0
�
siamese/scala3_1/cond/MergeMergesiamese/scala3_1/cond/Switch_3 siamese/scala3_1/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala3_1/cond/Merge_1Mergesiamese/scala3_1/cond/Switch_4 siamese/scala3_1/cond/Switch_2:1*
_output_shapes
	:�: *
T0*
N
e
 siamese/scala3_1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese/scala3_1/batchnorm/addAddsiamese/scala3_1/cond/Merge_1 siamese/scala3_1/batchnorm/add/y*
_output_shapes	
:�*
T0
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
 siamese/scala3_1/batchnorm/mul_2Mulsiamese/scala3_1/cond/Mergesiamese/scala3_1/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala3_1/batchnorm/subSubsiamese/scala3/bn/beta/read siamese/scala3_1/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese/scala3_1/batchnorm/add_1Add siamese/scala3_1/batchnorm/mul_1siamese/scala3_1/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese/scala3_1/ReluRelu siamese/scala3_1/batchnorm/add_1*'
_output_shapes
:�*
T0
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
d
"siamese/scala4_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/split_1Split"siamese/scala4_1/split_1/split_dim siamese/scala4/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese/scala4_1/Conv2DConv2Dsiamese/scala4_1/splitsiamese/scala4_1/split_1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala4_1/Conv2D_1Conv2Dsiamese/scala4_1/split:1siamese/scala4_1/split_1:1*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�*
T0
^
siamese/scala4_1/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala4_1/concatConcatV2siamese/scala4_1/Conv2Dsiamese/scala4_1/Conv2D_1siamese/scala4_1/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese/scala4_1/AddAddsiamese/scala4_1/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese/scala4_1/moments/Mean/reduction_indicesConst*
_output_shapes
:*!
valueB"          *
dtype0
�
siamese/scala4_1/moments/MeanMeansiamese/scala4_1/Add/siamese/scala4_1/moments/Mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese/scala4_1/moments/StopGradientStopGradientsiamese/scala4_1/moments/Mean*
T0*'
_output_shapes
:�
y
4siamese/scala4_1/moments/sufficient_statistics/ConstConst*
valueB
 *  �E*
dtype0*
_output_shapes
: 
�
2siamese/scala4_1/moments/sufficient_statistics/SubSubsiamese/scala4_1/Add%siamese/scala4_1/moments/StopGradient*'
_output_shapes
:�*
T0
�
@siamese/scala4_1/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese/scala4_1/Add%siamese/scala4_1/moments/StopGradient*
T0*'
_output_shapes
:�
�
Hsiamese/scala4_1/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese/scala4_1/moments/sufficient_statistics/mean_ssSum2siamese/scala4_1/moments/sufficient_statistics/SubHsiamese/scala4_1/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
Gsiamese/scala4_1/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese/scala4_1/moments/sufficient_statistics/var_ssSum@siamese/scala4_1/moments/sufficient_statistics/SquaredDifferenceGsiamese/scala4_1/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
i
siamese/scala4_1/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
 siamese/scala4_1/moments/ReshapeReshape%siamese/scala4_1/moments/StopGradientsiamese/scala4_1/moments/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
*siamese/scala4_1/moments/normalize/divisor
Reciprocal4siamese/scala4_1/moments/sufficient_statistics/Const7^siamese/scala4_1/moments/sufficient_statistics/mean_ss6^siamese/scala4_1/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
/siamese/scala4_1/moments/normalize/shifted_meanMul6siamese/scala4_1/moments/sufficient_statistics/mean_ss*siamese/scala4_1/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
'siamese/scala4_1/moments/normalize/meanAdd/siamese/scala4_1/moments/normalize/shifted_mean siamese/scala4_1/moments/Reshape*
_output_shapes	
:�*
T0
�
&siamese/scala4_1/moments/normalize/MulMul5siamese/scala4_1/moments/sufficient_statistics/var_ss*siamese/scala4_1/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
)siamese/scala4_1/moments/normalize/SquareSquare/siamese/scala4_1/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
+siamese/scala4_1/moments/normalize/varianceSub&siamese/scala4_1/moments/normalize/Mul)siamese/scala4_1/moments/normalize/Square*
T0*
_output_shapes	
:�
�
&siamese/scala4_1/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    *0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read'siamese/scala4_1/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese/scala4_1/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
ksiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Nsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Hsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
use_locking( *
T0
�
Csiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese/scala4_1/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Esiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
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
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstl^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Dsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
 siamese/scala4_1/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese/scala4_1/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
(siamese/scala4_1/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read+siamese/scala4_1/moments/normalize/variance*
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
usiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Tsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Nsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Isiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese/scala4_1/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Ksiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstv^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
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
Jsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
"siamese/scala4_1/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese/scala4_1/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
g
siamese/scala4_1/cond/SwitchSwitchis_training_1is_training_1*
T0
*
_output_shapes
: : 
k
siamese/scala4_1/cond/switch_tIdentitysiamese/scala4_1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese/scala4_1/cond/switch_fIdentitysiamese/scala4_1/cond/Switch*
T0
*
_output_shapes
: 
Y
siamese/scala4_1/cond/pred_idIdentityis_training_1*
_output_shapes
: *
T0

�
siamese/scala4_1/cond/Switch_1Switch'siamese/scala4_1/moments/normalize/meansiamese/scala4_1/cond/pred_id*
T0*:
_class0
.,loc:@siamese/scala4_1/moments/normalize/mean*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_2Switch+siamese/scala4_1/moments/normalize/variancesiamese/scala4_1/cond/pred_id*>
_class4
20loc:@siamese/scala4_1/moments/normalize/variance*"
_output_shapes
:�:�*
T0
�
%siamese/scala4_1/cond/Switch_3/SwitchSwitch"siamese/scala4/bn/moving_mean/readsiamese/scala4_1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_3Switch%siamese/scala4_1/cond/Switch_3/Switchsiamese/scala4_1/cond/pred_id*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�*
T0
�
%siamese/scala4_1/cond/Switch_4/SwitchSwitch&siamese/scala4/bn/moving_variance/readsiamese/scala4_1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese/scala4_1/cond/Switch_4Switch%siamese/scala4_1/cond/Switch_4/Switchsiamese/scala4_1/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese/scala4_1/cond/MergeMergesiamese/scala4_1/cond/Switch_3 siamese/scala4_1/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese/scala4_1/cond/Merge_1Mergesiamese/scala4_1/cond/Switch_4 siamese/scala4_1/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
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
 siamese/scala4_1/batchnorm/RsqrtRsqrtsiamese/scala4_1/batchnorm/add*
_output_shapes	
:�*
T0
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
 siamese/scala4_1/batchnorm/mul_2Mulsiamese/scala4_1/cond/Mergesiamese/scala4_1/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese/scala4_1/batchnorm/subSubsiamese/scala4/bn/beta/read siamese/scala4_1/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese/scala4_1/batchnorm/add_1Add siamese/scala4_1/batchnorm/mul_1siamese/scala4_1/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese/scala4_1/ReluRelu siamese/scala4_1/batchnorm/add_1*'
_output_shapes
:�*
T0
b
 siamese/scala5_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/splitSplit siamese/scala5_1/split/split_dimsiamese/scala4_1/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
d
"siamese/scala5_1/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese/scala5_1/split_1Split"siamese/scala5_1/split_1/split_dim siamese/scala5/conv/weights/read*
T0*<
_output_shapes*
(:��:��*
	num_split
�
siamese/scala5_1/Conv2DConv2Dsiamese/scala5_1/splitsiamese/scala5_1/split_1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese/scala5_1/Conv2D_1Conv2Dsiamese/scala5_1/split:1siamese/scala5_1/split_1:1*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
^
siamese/scala5_1/concat/axisConst*
dtype0*
_output_shapes
: *
value	B :
�
siamese/scala5_1/concatConcatV2siamese/scala5_1/Conv2Dsiamese/scala5_1/Conv2D_1siamese/scala5_1/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese/scala5_1/AddAddsiamese/scala5_1/concatsiamese/scala5/conv/biases/read*
T0*'
_output_shapes
:�
m
score/transpose/permConst*
_output_shapes
:*%
valueB"             *
dtype0
�
score/transpose	Transposesiamese/scala5/Addscore/transpose/perm*
Tperm0*
T0*'
_output_shapes
:�
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
Y
score/split_1/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
score/split_1Splitscore/split_1/split_dimsiamese/scala5_1/Add*�
_output_shapes�
�:�:�:�:�:�:�:�:�*
	num_split*
T0
�
score/Conv2DConv2Dscore/split_1score/split*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score/Conv2D_1Conv2Dscore/split_1:1score/split:1*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score/Conv2D_2Conv2Dscore/split_1:2score/split:2*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
T0*
data_formatNHWC*
strides

�
score/Conv2D_3Conv2Dscore/split_1:3score/split:3*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
T0*
data_formatNHWC*
strides

�
score/Conv2D_4Conv2Dscore/split_1:4score/split:4*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
T0*
data_formatNHWC*
strides

�
score/Conv2D_5Conv2Dscore/split_1:5score/split:5*
paddingVALID*&
_output_shapes
:*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
score/Conv2D_6Conv2Dscore/split_1:6score/split:6*&
_output_shapes
:*
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
:*
T0*
data_formatNHWC*
strides
*
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
score/transpose_1/permConst*
dtype0*
_output_shapes
:*%
valueB"             
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
VariableV2*&
_output_shapes
:*
shared_name *!
_class
loc:@adjust/weights*
	container *
shape:*
dtype0
�
adjust/weights/AssignAssignadjust/weights adjust/weights/Initializer/Const*
validate_shape(*&
_output_shapes
:*
use_locking(*
T0*!
_class
loc:@adjust/weights
�
adjust/weights/readIdentityadjust/weights*&
_output_shapes
:*
T0*!
_class
loc:@adjust/weights
�
/adjust/weights/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:*!
_class
loc:@adjust/weights*
dtype0*
_output_shapes
: 
�
0adjust/weights/Regularizer/l2_regularizer/L2LossL2Lossadjust/weights/read*
T0*!
_class
loc:@adjust/weights*
_output_shapes
: 
�
)adjust/weights/Regularizer/l2_regularizerMul/adjust/weights/Regularizer/l2_regularizer/scale0adjust/weights/Regularizer/l2_regularizer/L2Loss*!
_class
loc:@adjust/weights*
_output_shapes
: *
T0
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
adjust/biases/AssignAssignadjust/biasesadjust/biases/Initializer/Const*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@adjust/biases
t
adjust/biases/readIdentityadjust/biases*
_output_shapes
:*
T0* 
_class
loc:@adjust/biases
�
.adjust/biases/Regularizer/l2_regularizer/scaleConst*
valueB
 *o:* 
_class
loc:@adjust/biases*
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
:*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
e

adjust/AddAddadjust/Conv2Dadjust/biases/read*&
_output_shapes
:*
T0
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�,Badjust/biasesBadjust/weightsBsiamese/scala1/bn/betaBsiamese/scala1/bn/gammaBsiamese/scala1/bn/moving_meanB!siamese/scala1/bn/moving_varianceBsiamese/scala1/conv/biasesBsiamese/scala1/conv/weightsB3siamese/scala1/siamese/scala1/bn/moving_mean/biasedB7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepB7siamese/scala1/siamese/scala1/bn/moving_variance/biasedB;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepBsiamese/scala2/bn/betaBsiamese/scala2/bn/gammaBsiamese/scala2/bn/moving_meanB!siamese/scala2/bn/moving_varianceBsiamese/scala2/conv/biasesBsiamese/scala2/conv/weightsB3siamese/scala2/siamese/scala2/bn/moving_mean/biasedB7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepB7siamese/scala2/siamese/scala2/bn/moving_variance/biasedB;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepBsiamese/scala3/bn/betaBsiamese/scala3/bn/gammaBsiamese/scala3/bn/moving_meanB!siamese/scala3/bn/moving_varianceBsiamese/scala3/conv/biasesBsiamese/scala3/conv/weightsB3siamese/scala3/siamese/scala3/bn/moving_mean/biasedB7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepB7siamese/scala3/siamese/scala3/bn/moving_variance/biasedB;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepBsiamese/scala4/bn/betaBsiamese/scala4/bn/gammaBsiamese/scala4/bn/moving_meanB!siamese/scala4/bn/moving_varianceBsiamese/scala4/conv/biasesBsiamese/scala4/conv/weightsB3siamese/scala4/siamese/scala4/bn/moving_mean/biasedB7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepB7siamese/scala4/siamese/scala4/bn/moving_variance/biasedB;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepBsiamese/scala5/conv/biasesBsiamese/scala5/conv/weights*
dtype0*
_output_shapes
:,
�
save/SaveV2/shape_and_slicesConst*
_output_shapes
:,*k
valuebB`,B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
�
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesadjust/biasesadjust/weightssiamese/scala1/bn/betasiamese/scala1/bn/gammasiamese/scala1/bn/moving_mean!siamese/scala1/bn/moving_variancesiamese/scala1/conv/biasessiamese/scala1/conv/weights3siamese/scala1/siamese/scala1/bn/moving_mean/biased7siamese/scala1/siamese/scala1/bn/moving_mean/local_step7siamese/scala1/siamese/scala1/bn/moving_variance/biased;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsiamese/scala2/bn/betasiamese/scala2/bn/gammasiamese/scala2/bn/moving_mean!siamese/scala2/bn/moving_variancesiamese/scala2/conv/biasessiamese/scala2/conv/weights3siamese/scala2/siamese/scala2/bn/moving_mean/biased7siamese/scala2/siamese/scala2/bn/moving_mean/local_step7siamese/scala2/siamese/scala2/bn/moving_variance/biased;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsiamese/scala3/bn/betasiamese/scala3/bn/gammasiamese/scala3/bn/moving_mean!siamese/scala3/bn/moving_variancesiamese/scala3/conv/biasessiamese/scala3/conv/weights3siamese/scala3/siamese/scala3/bn/moving_mean/biased7siamese/scala3/siamese/scala3/bn/moving_mean/local_step7siamese/scala3/siamese/scala3/bn/moving_variance/biased;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsiamese/scala4/bn/betasiamese/scala4/bn/gammasiamese/scala4/bn/moving_mean!siamese/scala4/bn/moving_variancesiamese/scala4/conv/biasessiamese/scala4/conv/weights3siamese/scala4/siamese/scala4/bn/moving_mean/biased7siamese/scala4/siamese/scala4/bn/moving_mean/local_step7siamese/scala4/siamese/scala4/bn/moving_variance/biased;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsiamese/scala5/conv/biasessiamese/scala5/conv/weights*:
dtypes0
.2,
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
q
save/RestoreV2/tensor_namesConst*"
valueBBadjust/biases*
dtype0*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/AssignAssignadjust/biasessave/RestoreV2*
validate_shape(*
_output_shapes
:*
use_locking(*
T0* 
_class
loc:@adjust/biases
t
save/RestoreV2_1/tensor_namesConst*
dtype0*
_output_shapes
:*#
valueBBadjust/weights
j
!save/RestoreV2_1/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_1Assignadjust/weightssave/RestoreV2_1*
use_locking(*
T0*!
_class
loc:@adjust/weights*
validate_shape(*&
_output_shapes
:
|
save/RestoreV2_2/tensor_namesConst*+
value"B Bsiamese/scala1/bn/beta*
dtype0*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_2Assignsiamese/scala1/bn/betasave/RestoreV2_2*
use_locking(*
T0*)
_class
loc:@siamese/scala1/bn/beta*
validate_shape(*
_output_shapes
:`
}
save/RestoreV2_3/tensor_namesConst*,
value#B!Bsiamese/scala1/bn/gamma*
dtype0*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_3Assignsiamese/scala1/bn/gammasave/RestoreV2_3*
T0**
_class 
loc:@siamese/scala1/bn/gamma*
validate_shape(*
_output_shapes
:`*
use_locking(
�
save/RestoreV2_4/tensor_namesConst*2
value)B'Bsiamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_4Assignsiamese/scala1/bn/moving_meansave/RestoreV2_4*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`*
use_locking(*
T0
�
save/RestoreV2_5/tensor_namesConst*6
value-B+B!siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_5Assign!siamese/scala1/bn/moving_variancesave/RestoreV2_5*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
:`
�
save/RestoreV2_6/tensor_namesConst*/
value&B$Bsiamese/scala1/conv/biases*
dtype0*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_6Assignsiamese/scala1/conv/biasessave/RestoreV2_6*
use_locking(*
T0*-
_class#
!loc:@siamese/scala1/conv/biases*
validate_shape(*
_output_shapes
:`
�
save/RestoreV2_7/tensor_namesConst*0
value'B%Bsiamese/scala1/conv/weights*
dtype0*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_7Assignsiamese/scala1/conv/weightssave/RestoreV2_7*
T0*.
_class$
" loc:@siamese/scala1/conv/weights*
validate_shape(*&
_output_shapes
:`*
use_locking(
�
save/RestoreV2_8/tensor_namesConst*H
value?B=B3siamese/scala1/siamese/scala1/bn/moving_mean/biased*
dtype0*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_8Assign3siamese/scala1/siamese/scala1/bn/moving_mean/biasedsave/RestoreV2_8*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
:`
�
save/RestoreV2_9/tensor_namesConst*L
valueCBAB7siamese/scala1/siamese/scala1/bn/moving_mean/local_step*
dtype0*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_9Assign7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepsave/RestoreV2_9*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/RestoreV2_10/tensor_namesConst*L
valueCBAB7siamese/scala1/siamese/scala1/bn/moving_variance/biased*
dtype0*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_10Assign7siamese/scala1/siamese/scala1/bn/moving_variance/biasedsave/RestoreV2_10*
_output_shapes
:`*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(
�
save/RestoreV2_11/tensor_namesConst*P
valueGBEB;siamese/scala1/siamese/scala1/bn/moving_variance/local_step*
dtype0*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_11Assign;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepsave/RestoreV2_11*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
}
save/RestoreV2_12/tensor_namesConst*+
value"B Bsiamese/scala2/bn/beta*
dtype0*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_12Assignsiamese/scala2/bn/betasave/RestoreV2_12*
use_locking(*
T0*)
_class
loc:@siamese/scala2/bn/beta*
validate_shape(*
_output_shapes	
:�
~
save/RestoreV2_13/tensor_namesConst*,
value#B!Bsiamese/scala2/bn/gamma*
dtype0*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_13Assignsiamese/scala2/bn/gammasave/RestoreV2_13*
use_locking(*
T0**
_class 
loc:@siamese/scala2/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_14/tensor_namesConst*2
value)B'Bsiamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_14Assignsiamese/scala2/bn/moving_meansave/RestoreV2_14*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(
�
save/RestoreV2_15/tensor_namesConst*6
value-B+B!siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_15Assign!siamese/scala2/bn/moving_variancesave/RestoreV2_15*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/RestoreV2_16/tensor_namesConst*/
value&B$Bsiamese/scala2/conv/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_16Assignsiamese/scala2/conv/biasessave/RestoreV2_16*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala2/conv/biases
�
save/RestoreV2_17/tensor_namesConst*
dtype0*
_output_shapes
:*0
value'B%Bsiamese/scala2/conv/weights
k
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_17Assignsiamese/scala2/conv/weightssave/RestoreV2_17*
use_locking(*
T0*.
_class$
" loc:@siamese/scala2/conv/weights*
validate_shape(*'
_output_shapes
:0�
�
save/RestoreV2_18/tensor_namesConst*H
value?B=B3siamese/scala2/siamese/scala2/bn/moving_mean/biased*
dtype0*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_18Assign3siamese/scala2/siamese/scala2/bn/moving_mean/biasedsave/RestoreV2_18*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
save/RestoreV2_19/tensor_namesConst*
_output_shapes
:*L
valueCBAB7siamese/scala2/siamese/scala2/bn/moving_mean/local_step*
dtype0
k
"save/RestoreV2_19/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_19Assign7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepsave/RestoreV2_19*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/RestoreV2_20/tensor_namesConst*L
valueCBAB7siamese/scala2/siamese/scala2/bn/moving_variance/biased*
dtype0*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_20Assign7siamese/scala2/siamese/scala2/bn/moving_variance/biasedsave/RestoreV2_20*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/RestoreV2_21/tensor_namesConst*P
valueGBEB;siamese/scala2/siamese/scala2/bn/moving_variance/local_step*
dtype0*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_21Assign;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepsave/RestoreV2_21*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
}
save/RestoreV2_22/tensor_namesConst*
dtype0*
_output_shapes
:*+
value"B Bsiamese/scala3/bn/beta
k
"save/RestoreV2_22/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_22Assignsiamese/scala3/bn/betasave/RestoreV2_22*
_output_shapes	
:�*
use_locking(*
T0*)
_class
loc:@siamese/scala3/bn/beta*
validate_shape(
~
save/RestoreV2_23/tensor_namesConst*,
value#B!Bsiamese/scala3/bn/gamma*
dtype0*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_23Assignsiamese/scala3/bn/gammasave/RestoreV2_23*
_output_shapes	
:�*
use_locking(*
T0**
_class 
loc:@siamese/scala3/bn/gamma*
validate_shape(
�
save/RestoreV2_24/tensor_namesConst*
_output_shapes
:*2
value)B'Bsiamese/scala3/bn/moving_mean*
dtype0
k
"save/RestoreV2_24/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_24Assignsiamese/scala3/bn/moving_meansave/RestoreV2_24*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_25/tensor_namesConst*
_output_shapes
:*6
value-B+B!siamese/scala3/bn/moving_variance*
dtype0
k
"save/RestoreV2_25/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_25Assign!siamese/scala3/bn/moving_variancesave/RestoreV2_25*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_26/tensor_namesConst*/
value&B$Bsiamese/scala3/conv/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_26Assignsiamese/scala3/conv/biasessave/RestoreV2_26*
use_locking(*
T0*-
_class#
!loc:@siamese/scala3/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_27/tensor_namesConst*0
value'B%Bsiamese/scala3/conv/weights*
dtype0*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_27Assignsiamese/scala3/conv/weightssave/RestoreV2_27*.
_class$
" loc:@siamese/scala3/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(*
T0
�
save/RestoreV2_28/tensor_namesConst*H
value?B=B3siamese/scala3/siamese/scala3/bn/moving_mean/biased*
dtype0*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_28Assign3siamese/scala3/siamese/scala3/bn/moving_mean/biasedsave/RestoreV2_28*
_output_shapes	
:�*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(
�
save/RestoreV2_29/tensor_namesConst*L
valueCBAB7siamese/scala3/siamese/scala3/bn/moving_mean/local_step*
dtype0*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_29Assign7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepsave/RestoreV2_29*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save/RestoreV2_30/tensor_namesConst*L
valueCBAB7siamese/scala3/siamese/scala3/bn/moving_variance/biased*
dtype0*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_30Assign7siamese/scala3/siamese/scala3/bn/moving_variance/biasedsave/RestoreV2_30*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0
�
save/RestoreV2_31/tensor_namesConst*
dtype0*
_output_shapes
:*P
valueGBEB;siamese/scala3/siamese/scala3/bn/moving_variance/local_step
k
"save/RestoreV2_31/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_31Assign;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepsave/RestoreV2_31*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
validate_shape(*
_output_shapes
: *
use_locking(
}
save/RestoreV2_32/tensor_namesConst*+
value"B Bsiamese/scala4/bn/beta*
dtype0*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
_output_shapes
:*
valueB
B *
dtype0
�
save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_32Assignsiamese/scala4/bn/betasave/RestoreV2_32*
T0*)
_class
loc:@siamese/scala4/bn/beta*
validate_shape(*
_output_shapes	
:�*
use_locking(
~
save/RestoreV2_33/tensor_namesConst*,
value#B!Bsiamese/scala4/bn/gamma*
dtype0*
_output_shapes
:
k
"save/RestoreV2_33/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_33	RestoreV2
save/Constsave/RestoreV2_33/tensor_names"save/RestoreV2_33/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_33Assignsiamese/scala4/bn/gammasave/RestoreV2_33*
use_locking(*
T0**
_class 
loc:@siamese/scala4/bn/gamma*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_34/tensor_namesConst*
dtype0*
_output_shapes
:*2
value)B'Bsiamese/scala4/bn/moving_mean
k
"save/RestoreV2_34/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_34	RestoreV2
save/Constsave/RestoreV2_34/tensor_names"save/RestoreV2_34/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_34Assignsiamese/scala4/bn/moving_meansave/RestoreV2_34*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_35/tensor_namesConst*6
value-B+B!siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
:
k
"save/RestoreV2_35/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_35	RestoreV2
save/Constsave/RestoreV2_35/tensor_names"save/RestoreV2_35/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_35Assign!siamese/scala4/bn/moving_variancesave/RestoreV2_35*
validate_shape(*
_output_shapes	
:�*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
save/RestoreV2_36/tensor_namesConst*/
value&B$Bsiamese/scala4/conv/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_36/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_36	RestoreV2
save/Constsave/RestoreV2_36/tensor_names"save/RestoreV2_36/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_36Assignsiamese/scala4/conv/biasessave/RestoreV2_36*
_output_shapes	
:�*
use_locking(*
T0*-
_class#
!loc:@siamese/scala4/conv/biases*
validate_shape(
�
save/RestoreV2_37/tensor_namesConst*0
value'B%Bsiamese/scala4/conv/weights*
dtype0*
_output_shapes
:
k
"save/RestoreV2_37/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_37	RestoreV2
save/Constsave/RestoreV2_37/tensor_names"save/RestoreV2_37/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_37Assignsiamese/scala4/conv/weightssave/RestoreV2_37*
T0*.
_class$
" loc:@siamese/scala4/conv/weights*
validate_shape(*(
_output_shapes
:��*
use_locking(
�
save/RestoreV2_38/tensor_namesConst*H
value?B=B3siamese/scala4/siamese/scala4/bn/moving_mean/biased*
dtype0*
_output_shapes
:
k
"save/RestoreV2_38/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_38	RestoreV2
save/Constsave/RestoreV2_38/tensor_names"save/RestoreV2_38/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_38Assign3siamese/scala4/siamese/scala4/bn/moving_mean/biasedsave/RestoreV2_38*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_39/tensor_namesConst*L
valueCBAB7siamese/scala4/siamese/scala4/bn/moving_mean/local_step*
dtype0*
_output_shapes
:
k
"save/RestoreV2_39/shape_and_slicesConst*
dtype0*
_output_shapes
:*
valueB
B 
�
save/RestoreV2_39	RestoreV2
save/Constsave/RestoreV2_39/tensor_names"save/RestoreV2_39/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_39Assign7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepsave/RestoreV2_39*
use_locking(*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
validate_shape(*
_output_shapes
: 
�
save/RestoreV2_40/tensor_namesConst*L
valueCBAB7siamese/scala4/siamese/scala4/bn/moving_variance/biased*
dtype0*
_output_shapes
:
k
"save/RestoreV2_40/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_40	RestoreV2
save/Constsave/RestoreV2_40/tensor_names"save/RestoreV2_40/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_40Assign7siamese/scala4/siamese/scala4/bn/moving_variance/biasedsave/RestoreV2_40*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_41/tensor_namesConst*P
valueGBEB;siamese/scala4/siamese/scala4/bn/moving_variance/local_step*
dtype0*
_output_shapes
:
k
"save/RestoreV2_41/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_41	RestoreV2
save/Constsave/RestoreV2_41/tensor_names"save/RestoreV2_41/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_41Assign;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepsave/RestoreV2_41*
use_locking(*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
validate_shape(*
_output_shapes
: 
�
save/RestoreV2_42/tensor_namesConst*/
value&B$Bsiamese/scala5/conv/biases*
dtype0*
_output_shapes
:
k
"save/RestoreV2_42/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_42	RestoreV2
save/Constsave/RestoreV2_42/tensor_names"save/RestoreV2_42/shape_and_slices*
dtypes
2*
_output_shapes
:
�
save/Assign_42Assignsiamese/scala5/conv/biasessave/RestoreV2_42*
use_locking(*
T0*-
_class#
!loc:@siamese/scala5/conv/biases*
validate_shape(*
_output_shapes	
:�
�
save/RestoreV2_43/tensor_namesConst*0
value'B%Bsiamese/scala5/conv/weights*
dtype0*
_output_shapes
:
k
"save/RestoreV2_43/shape_and_slicesConst*
valueB
B *
dtype0*
_output_shapes
:
�
save/RestoreV2_43	RestoreV2
save/Constsave/RestoreV2_43/tensor_names"save/RestoreV2_43/shape_and_slices*
_output_shapes
:*
dtypes
2
�
save/Assign_43Assignsiamese/scala5/conv/weightssave/RestoreV2_43*
use_locking(*
T0*.
_class$
" loc:@siamese/scala5/conv/weights*
validate_shape(*(
_output_shapes
:��
�
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43
�
siamese_1/scala1/Conv2DConv2DPlaceholder siamese/scala1/conv/weights/read*
paddingVALID*&
_output_shapes
:;;`*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese_1/scala1/AddAddsiamese_1/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:;;`
�
/siamese_1/scala1/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala1/moments/MeanMeansiamese_1/scala1/Add/siamese_1/scala1/moments/Mean/reduction_indices*&
_output_shapes
:`*
	keep_dims(*

Tidx0*
T0
�
%siamese_1/scala1/moments/StopGradientStopGradientsiamese_1/scala1/moments/Mean*&
_output_shapes
:`*
T0
y
4siamese_1/scala1/moments/sufficient_statistics/ConstConst*
valueB
 * �YE*
dtype0*
_output_shapes
: 
�
2siamese_1/scala1/moments/sufficient_statistics/SubSubsiamese_1/scala1/Add%siamese_1/scala1/moments/StopGradient*
T0*&
_output_shapes
:;;`
�
@siamese_1/scala1/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese_1/scala1/Add%siamese_1/scala1/moments/StopGradient*&
_output_shapes
:;;`*
T0
�
Hsiamese_1/scala1/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese_1/scala1/moments/sufficient_statistics/mean_ssSum2siamese_1/scala1/moments/sufficient_statistics/SubHsiamese_1/scala1/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes
:`*
	keep_dims( *

Tidx0*
T0
�
Gsiamese_1/scala1/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese_1/scala1/moments/sufficient_statistics/var_ssSum@siamese_1/scala1/moments/sufficient_statistics/SquaredDifferenceGsiamese_1/scala1/moments/sufficient_statistics/var_ss/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:`
h
siamese_1/scala1/moments/ShapeConst*
dtype0*
_output_shapes
:*
valueB:`
�
 siamese_1/scala1/moments/ReshapeReshape%siamese_1/scala1/moments/StopGradientsiamese_1/scala1/moments/Shape*
T0*
Tshape0*
_output_shapes
:`
�
*siamese_1/scala1/moments/normalize/divisor
Reciprocal4siamese_1/scala1/moments/sufficient_statistics/Const7^siamese_1/scala1/moments/sufficient_statistics/mean_ss6^siamese_1/scala1/moments/sufficient_statistics/var_ss*
_output_shapes
: *
T0
�
/siamese_1/scala1/moments/normalize/shifted_meanMul6siamese_1/scala1/moments/sufficient_statistics/mean_ss*siamese_1/scala1/moments/normalize/divisor*
T0*
_output_shapes
:`
�
'siamese_1/scala1/moments/normalize/meanAdd/siamese_1/scala1/moments/normalize/shifted_mean siamese_1/scala1/moments/Reshape*
_output_shapes
:`*
T0
�
&siamese_1/scala1/moments/normalize/MulMul5siamese_1/scala1/moments/sufficient_statistics/var_ss*siamese_1/scala1/moments/normalize/divisor*
_output_shapes
:`*
T0
�
)siamese_1/scala1/moments/normalize/SquareSquare/siamese_1/scala1/moments/normalize/shifted_mean*
T0*
_output_shapes
:`
�
+siamese_1/scala1/moments/normalize/varianceSub&siamese_1/scala1/moments/normalize/Mul)siamese_1/scala1/moments/normalize/Square*
_output_shapes
:`*
T0
�
&siamese_1/scala1/AssignMovingAvg/decayConst*
_output_shapes
: *
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0
�
Dsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/ConstConst*
valueB`*    *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
:`
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read'siamese_1/scala1/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_1/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
Nsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Hsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepNsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Csiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
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
Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Bsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstl^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese_1/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
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
(siamese_1/scala1/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/ConstConst*
valueB`*    *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
:`
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read+siamese_1/scala1/moments/normalize/variance*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Hsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mulMulHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub(siamese_1/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
usiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( 
�
Tsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
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
Isiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
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
Ksiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
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
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstv^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2SubLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xHsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese_1/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_1/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`*
use_locking( *
T0
c
siamese_1/scala1/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala1/cond/switch_tIdentitysiamese_1/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala1/cond/switch_fIdentitysiamese_1/scala1/cond/Switch*
_output_shapes
: *
T0

W
siamese_1/scala1/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala1/cond/Switch_1Switch'siamese_1/scala1/moments/normalize/meansiamese_1/scala1/cond/pred_id*
T0*:
_class0
.,loc:@siamese_1/scala1/moments/normalize/mean* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_2Switch+siamese_1/scala1/moments/normalize/variancesiamese_1/scala1/cond/pred_id*
T0*>
_class4
20loc:@siamese_1/scala1/moments/normalize/variance* 
_output_shapes
:`:`
�
%siamese_1/scala1/cond/Switch_3/SwitchSwitch"siamese/scala1/bn/moving_mean/readsiamese_1/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_3Switch%siamese_1/scala1/cond/Switch_3/Switchsiamese_1/scala1/cond/pred_id*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`*
T0
�
%siamese_1/scala1/cond/Switch_4/SwitchSwitch&siamese/scala1/bn/moving_variance/readsiamese_1/scala1/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`
�
siamese_1/scala1/cond/Switch_4Switch%siamese_1/scala1/cond/Switch_4/Switchsiamese_1/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
siamese_1/scala1/cond/MergeMergesiamese_1/scala1/cond/Switch_3 siamese_1/scala1/cond/Switch_1:1*
T0*
N*
_output_shapes

:`: 
�
siamese_1/scala1/cond/Merge_1Mergesiamese_1/scala1/cond/Switch_4 siamese_1/scala1/cond/Switch_2:1*
T0*
N*
_output_shapes

:`: 
e
 siamese_1/scala1/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala1/batchnorm/addAddsiamese_1/scala1/cond/Merge_1 siamese_1/scala1/batchnorm/add/y*
_output_shapes
:`*
T0
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
 siamese_1/scala1/batchnorm/mul_2Mulsiamese_1/scala1/cond/Mergesiamese_1/scala1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese_1/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_1/scala1/batchnorm/mul_2*
T0*
_output_shapes
:`
�
 siamese_1/scala1/batchnorm/add_1Add siamese_1/scala1/batchnorm/mul_1siamese_1/scala1/batchnorm/sub*
T0*&
_output_shapes
:;;`
p
siamese_1/scala1/ReluRelu siamese_1/scala1/batchnorm/add_1*&
_output_shapes
:;;`*
T0
�
siamese_1/scala1/poll/MaxPoolMaxPoolsiamese_1/scala1/Relu*&
_output_shapes
:`*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
b
 siamese_1/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/splitSplit siamese_1/scala2/split/split_dimsiamese_1/scala1/poll/MaxPool*
T0*8
_output_shapes&
$:0:0*
	num_split
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
siamese_1/scala2/Conv2DConv2Dsiamese_1/scala2/splitsiamese_1/scala2/split_1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_1/scala2/Conv2D_1Conv2Dsiamese_1/scala2/split:1siamese_1/scala2/split_1:1*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
^
siamese_1/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/concatConcatV2siamese_1/scala2/Conv2Dsiamese_1/scala2/Conv2D_1siamese_1/scala2/concat/axis*
N*'
_output_shapes
:�*

Tidx0*
T0
�
siamese_1/scala2/AddAddsiamese_1/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_1/scala2/moments/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_1/scala2/moments/MeanMeansiamese_1/scala2/Add/siamese_1/scala2/moments/Mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_1/scala2/moments/StopGradientStopGradientsiamese_1/scala2/moments/Mean*
T0*'
_output_shapes
:�
y
4siamese_1/scala2/moments/sufficient_statistics/ConstConst*
valueB
 * @D*
dtype0*
_output_shapes
: 
�
2siamese_1/scala2/moments/sufficient_statistics/SubSubsiamese_1/scala2/Add%siamese_1/scala2/moments/StopGradient*'
_output_shapes
:�*
T0
�
@siamese_1/scala2/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese_1/scala2/Add%siamese_1/scala2/moments/StopGradient*
T0*'
_output_shapes
:�
�
Hsiamese_1/scala2/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese_1/scala2/moments/sufficient_statistics/mean_ssSum2siamese_1/scala2/moments/sufficient_statistics/SubHsiamese_1/scala2/moments/sufficient_statistics/mean_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
Gsiamese_1/scala2/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese_1/scala2/moments/sufficient_statistics/var_ssSum@siamese_1/scala2/moments/sufficient_statistics/SquaredDifferenceGsiamese_1/scala2/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
i
siamese_1/scala2/moments/ShapeConst*
_output_shapes
:*
valueB:�*
dtype0
�
 siamese_1/scala2/moments/ReshapeReshape%siamese_1/scala2/moments/StopGradientsiamese_1/scala2/moments/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
*siamese_1/scala2/moments/normalize/divisor
Reciprocal4siamese_1/scala2/moments/sufficient_statistics/Const7^siamese_1/scala2/moments/sufficient_statistics/mean_ss6^siamese_1/scala2/moments/sufficient_statistics/var_ss*
_output_shapes
: *
T0
�
/siamese_1/scala2/moments/normalize/shifted_meanMul6siamese_1/scala2/moments/sufficient_statistics/mean_ss*siamese_1/scala2/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
'siamese_1/scala2/moments/normalize/meanAdd/siamese_1/scala2/moments/normalize/shifted_mean siamese_1/scala2/moments/Reshape*
T0*
_output_shapes	
:�
�
&siamese_1/scala2/moments/normalize/MulMul5siamese_1/scala2/moments/sufficient_statistics/var_ss*siamese_1/scala2/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
)siamese_1/scala2/moments/normalize/SquareSquare/siamese_1/scala2/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
+siamese_1/scala2/moments/normalize/varianceSub&siamese_1/scala2/moments/normalize/Mul)siamese_1/scala2/moments/normalize/Square*
_output_shapes	
:�*
T0
�
&siamese_1/scala2/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read'siamese_1/scala2/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mulMulBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub&siamese_1/scala2/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
ksiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
Nsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Hsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Csiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/x&siamese_1/scala2/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstl^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3Sub"siamese/scala2/bn/moving_mean/readFsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
 siamese_1/scala2/AssignMovingAvg	AssignSubsiamese/scala2/bn/moving_meanDsiamese_1/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
(siamese_1/scala2/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read+siamese_1/scala2/moments/normalize/variance*
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
usiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
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
Isiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_1/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/PowPowJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1Ksiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstv^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
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
"siamese_1/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_1/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
c
siamese_1/scala2/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_1/scala2/cond/switch_tIdentitysiamese_1/scala2/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_1/scala2/cond/switch_fIdentitysiamese_1/scala2/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_1/scala2/cond/Switch_1Switch'siamese_1/scala2/moments/normalize/meansiamese_1/scala2/cond/pred_id*
T0*:
_class0
.,loc:@siamese_1/scala2/moments/normalize/mean*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_2Switch+siamese_1/scala2/moments/normalize/variancesiamese_1/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*>
_class4
20loc:@siamese_1/scala2/moments/normalize/variance
�
%siamese_1/scala2/cond/Switch_3/SwitchSwitch"siamese/scala2/bn/moving_mean/readsiamese_1/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_3Switch%siamese_1/scala2/cond/Switch_3/Switchsiamese_1/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
%siamese_1/scala2/cond/Switch_4/SwitchSwitch&siamese/scala2/bn/moving_variance/readsiamese_1/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/Switch_4Switch%siamese_1/scala2/cond/Switch_4/Switchsiamese_1/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala2/cond/MergeMergesiamese_1/scala2/cond/Switch_3 siamese_1/scala2/cond/Switch_1:1*
N*
_output_shapes
	:�: *
T0
�
siamese_1/scala2/cond/Merge_1Mergesiamese_1/scala2/cond/Switch_4 siamese_1/scala2/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
e
 siamese_1/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala2/batchnorm/addAddsiamese_1/scala2/cond/Merge_1 siamese_1/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_1/scala2/batchnorm/RsqrtRsqrtsiamese_1/scala2/batchnorm/add*
_output_shapes	
:�*
T0
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
 siamese_1/scala2/batchnorm/add_1Add siamese_1/scala2/batchnorm/mul_1siamese_1/scala2/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_1/scala2/ReluRelu siamese_1/scala2/batchnorm/add_1*'
_output_shapes
:�*
T0
�
siamese_1/scala2/poll/MaxPoolMaxPoolsiamese_1/scala2/Relu*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
siamese_1/scala3/Conv2DConv2Dsiamese_1/scala2/poll/MaxPool siamese/scala3/conv/weights/read*
paddingVALID*'
_output_shapes
:

�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese_1/scala3/AddAddsiamese_1/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:

�
�
/siamese_1/scala3/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala3/moments/MeanMeansiamese_1/scala3/Add/siamese_1/scala3/moments/Mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:�
�
%siamese_1/scala3/moments/StopGradientStopGradientsiamese_1/scala3/moments/Mean*'
_output_shapes
:�*
T0
y
4siamese_1/scala3/moments/sufficient_statistics/ConstConst*
valueB
 *  �B*
dtype0*
_output_shapes
: 
�
2siamese_1/scala3/moments/sufficient_statistics/SubSubsiamese_1/scala3/Add%siamese_1/scala3/moments/StopGradient*'
_output_shapes
:

�*
T0
�
@siamese_1/scala3/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese_1/scala3/Add%siamese_1/scala3/moments/StopGradient*
T0*'
_output_shapes
:

�
�
Hsiamese_1/scala3/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
6siamese_1/scala3/moments/sufficient_statistics/mean_ssSum2siamese_1/scala3/moments/sufficient_statistics/SubHsiamese_1/scala3/moments/sufficient_statistics/mean_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
�
Gsiamese_1/scala3/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese_1/scala3/moments/sufficient_statistics/var_ssSum@siamese_1/scala3/moments/sufficient_statistics/SquaredDifferenceGsiamese_1/scala3/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
i
siamese_1/scala3/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
 siamese_1/scala3/moments/ReshapeReshape%siamese_1/scala3/moments/StopGradientsiamese_1/scala3/moments/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
*siamese_1/scala3/moments/normalize/divisor
Reciprocal4siamese_1/scala3/moments/sufficient_statistics/Const7^siamese_1/scala3/moments/sufficient_statistics/mean_ss6^siamese_1/scala3/moments/sufficient_statistics/var_ss*
_output_shapes
: *
T0
�
/siamese_1/scala3/moments/normalize/shifted_meanMul6siamese_1/scala3/moments/sufficient_statistics/mean_ss*siamese_1/scala3/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
'siamese_1/scala3/moments/normalize/meanAdd/siamese_1/scala3/moments/normalize/shifted_mean siamese_1/scala3/moments/Reshape*
T0*
_output_shapes	
:�
�
&siamese_1/scala3/moments/normalize/MulMul5siamese_1/scala3/moments/sufficient_statistics/var_ss*siamese_1/scala3/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
)siamese_1/scala3/moments/normalize/SquareSquare/siamese_1/scala3/moments/normalize/shifted_mean*
_output_shapes	
:�*
T0
�
+siamese_1/scala3/moments/normalize/varianceSub&siamese_1/scala3/moments/normalize/Mul)siamese_1/scala3/moments/normalize/Square*
T0*
_output_shapes	
:�
�
&siamese_1/scala3/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read'siamese_1/scala3/moments/normalize/mean*
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
ksiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
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
Csiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_1/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstl^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Fsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese_1/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_1/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
(siamese_1/scala3/AssignMovingAvg_1/decayConst*
_output_shapes
: *
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read+siamese_1/scala3/moments/normalize/variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_1/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Tsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Nsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( 
�
Isiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
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
Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstv^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2SubLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xHsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truedivRealDivIsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese_1/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_1/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
c
siamese_1/scala3/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala3/cond/switch_tIdentitysiamese_1/scala3/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_1/scala3/cond/switch_fIdentitysiamese_1/scala3/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala3/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala3/cond/Switch_1Switch'siamese_1/scala3/moments/normalize/meansiamese_1/scala3/cond/pred_id*
T0*:
_class0
.,loc:@siamese_1/scala3/moments/normalize/mean*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/Switch_2Switch+siamese_1/scala3/moments/normalize/variancesiamese_1/scala3/cond/pred_id*
T0*>
_class4
20loc:@siamese_1/scala3/moments/normalize/variance*"
_output_shapes
:�:�
�
%siamese_1/scala3/cond/Switch_3/SwitchSwitch"siamese/scala3/bn/moving_mean/readsiamese_1/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/Switch_3Switch%siamese_1/scala3/cond/Switch_3/Switchsiamese_1/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
%siamese_1/scala3/cond/Switch_4/SwitchSwitch&siamese/scala3/bn/moving_variance/readsiamese_1/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese_1/scala3/cond/Switch_4Switch%siamese_1/scala3/cond/Switch_4/Switchsiamese_1/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala3/cond/MergeMergesiamese_1/scala3/cond/Switch_3 siamese_1/scala3/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_1/scala3/cond/Merge_1Mergesiamese_1/scala3/cond/Switch_4 siamese_1/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_1/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_1/scala3/batchnorm/addAddsiamese_1/scala3/cond/Merge_1 siamese_1/scala3/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_1/scala3/batchnorm/RsqrtRsqrtsiamese_1/scala3/batchnorm/add*
_output_shapes	
:�*
T0
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
 siamese_1/scala3/batchnorm/mul_2Mulsiamese_1/scala3/cond/Mergesiamese_1/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_1/scala3/batchnorm/mul_2*
T0*
_output_shapes	
:�
�
 siamese_1/scala3/batchnorm/add_1Add siamese_1/scala3/batchnorm/mul_1siamese_1/scala3/batchnorm/sub*'
_output_shapes
:

�*
T0
q
siamese_1/scala3/ReluRelu siamese_1/scala3/batchnorm/add_1*
T0*'
_output_shapes
:

�
b
 siamese_1/scala4/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/splitSplit siamese_1/scala4/split/split_dimsiamese_1/scala3/Relu*:
_output_shapes(
&:

�:

�*
	num_split*
T0
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
siamese_1/scala4/Conv2DConv2Dsiamese_1/scala4/splitsiamese_1/scala4/split_1*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
�
siamese_1/scala4/Conv2D_1Conv2Dsiamese_1/scala4/split:1siamese_1/scala4/split_1:1*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
^
siamese_1/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala4/concatConcatV2siamese_1/scala4/Conv2Dsiamese_1/scala4/Conv2D_1siamese_1/scala4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_1/scala4/AddAddsiamese_1/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_1/scala4/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_1/scala4/moments/MeanMeansiamese_1/scala4/Add/siamese_1/scala4/moments/Mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_1/scala4/moments/StopGradientStopGradientsiamese_1/scala4/moments/Mean*
T0*'
_output_shapes
:�
y
4siamese_1/scala4/moments/sufficient_statistics/ConstConst*
valueB
 *  �B*
dtype0*
_output_shapes
: 
�
2siamese_1/scala4/moments/sufficient_statistics/SubSubsiamese_1/scala4/Add%siamese_1/scala4/moments/StopGradient*'
_output_shapes
:�*
T0
�
@siamese_1/scala4/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese_1/scala4/Add%siamese_1/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
�
Hsiamese_1/scala4/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
6siamese_1/scala4/moments/sufficient_statistics/mean_ssSum2siamese_1/scala4/moments/sufficient_statistics/SubHsiamese_1/scala4/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
Gsiamese_1/scala4/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese_1/scala4/moments/sufficient_statistics/var_ssSum@siamese_1/scala4/moments/sufficient_statistics/SquaredDifferenceGsiamese_1/scala4/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
i
siamese_1/scala4/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
 siamese_1/scala4/moments/ReshapeReshape%siamese_1/scala4/moments/StopGradientsiamese_1/scala4/moments/Shape*
_output_shapes	
:�*
T0*
Tshape0
�
*siamese_1/scala4/moments/normalize/divisor
Reciprocal4siamese_1/scala4/moments/sufficient_statistics/Const7^siamese_1/scala4/moments/sufficient_statistics/mean_ss6^siamese_1/scala4/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
/siamese_1/scala4/moments/normalize/shifted_meanMul6siamese_1/scala4/moments/sufficient_statistics/mean_ss*siamese_1/scala4/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
'siamese_1/scala4/moments/normalize/meanAdd/siamese_1/scala4/moments/normalize/shifted_mean siamese_1/scala4/moments/Reshape*
_output_shapes	
:�*
T0
�
&siamese_1/scala4/moments/normalize/MulMul5siamese_1/scala4/moments/sufficient_statistics/var_ss*siamese_1/scala4/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
)siamese_1/scala4/moments/normalize/SquareSquare/siamese_1/scala4/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
+siamese_1/scala4/moments/normalize/varianceSub&siamese_1/scala4/moments/normalize/Mul)siamese_1/scala4/moments/normalize/Square*
_output_shapes	
:�*
T0
�
&siamese_1/scala4/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes	
:�
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read'siamese_1/scala4/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_1/scala4/AssignMovingAvg/decay*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
ksiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
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
Csiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
T0
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/x&siamese_1/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/PowPowDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1Esiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstl^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truedivRealDivCsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3Sub"siamese/scala4/bn/moving_mean/readFsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/truediv*
_output_shapes	
:�*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
 siamese_1/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_1/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
(siamese_1/scala4/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read+siamese_1/scala4/moments/normalize/variance*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_1/scala4/AssignMovingAvg_1/decay*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
usiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Nsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Isiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/x(siamese_1/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Ksiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/PowPowJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1Ksiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstv^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
"siamese_1/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_1/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
_output_shapes	
:�*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
c
siamese_1/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_1/scala4/cond/switch_tIdentitysiamese_1/scala4/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_1/scala4/cond/switch_fIdentitysiamese_1/scala4/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_1/scala4/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_1/scala4/cond/Switch_1Switch'siamese_1/scala4/moments/normalize/meansiamese_1/scala4/cond/pred_id*
T0*:
_class0
.,loc:@siamese_1/scala4/moments/normalize/mean*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_2Switch+siamese_1/scala4/moments/normalize/variancesiamese_1/scala4/cond/pred_id*
T0*>
_class4
20loc:@siamese_1/scala4/moments/normalize/variance*"
_output_shapes
:�:�
�
%siamese_1/scala4/cond/Switch_3/SwitchSwitch"siamese/scala4/bn/moving_mean/readsiamese_1/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_3Switch%siamese_1/scala4/cond/Switch_3/Switchsiamese_1/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
%siamese_1/scala4/cond/Switch_4/SwitchSwitch&siamese/scala4/bn/moving_variance/readsiamese_1/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/Switch_4Switch%siamese_1/scala4/cond/Switch_4/Switchsiamese_1/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_1/scala4/cond/MergeMergesiamese_1/scala4/cond/Switch_3 siamese_1/scala4/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_1/scala4/cond/Merge_1Mergesiamese_1/scala4/cond/Switch_4 siamese_1/scala4/cond/Switch_2:1*
N*
_output_shapes
	:�: *
T0
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
siamese_1/scala4/batchnorm/mulMul siamese_1/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
T0*
_output_shapes	
:�
�
 siamese_1/scala4/batchnorm/mul_1Mulsiamese_1/scala4/Addsiamese_1/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_1/scala4/batchnorm/mul_2Mulsiamese_1/scala4/cond/Mergesiamese_1/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_1/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_1/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_1/scala4/batchnorm/add_1Add siamese_1/scala4/batchnorm/mul_1siamese_1/scala4/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_1/scala4/ReluRelu siamese_1/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
b
 siamese_1/scala5/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala5/splitSplit siamese_1/scala5/split/split_dimsiamese_1/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
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
siamese_1/scala5/Conv2DConv2Dsiamese_1/scala5/splitsiamese_1/scala5/split_1*
paddingVALID*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(
�
siamese_1/scala5/Conv2D_1Conv2Dsiamese_1/scala5/split:1siamese_1/scala5/split_1:1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_1/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_1/scala5/concatConcatV2siamese_1/scala5/Conv2Dsiamese_1/scala5/Conv2D_1siamese_1/scala5/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_1/scala5/AddAddsiamese_1/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
�
ConstConst*��
value��B���"��P�<��:=�=������L:(�`�Q���,�$��<��n�0,���O< m���O=F,!�	5=�����=@M�:���j��@��<�O=Xb����߼�݈����\���i�=`y�;�<�=$v!�h�z<�����Q���ֽ0����24=~��=Ү=� �9 �)8H'`<�:P�R=��5�T=�z~=�#>x�<��� ԅ=�MD<ۃ=/�*=�4���@W
�T��< ���p-m��,��m�=�|=Ѳ����0�p<P���ܱ���Z�<3Te=h]<���)<=�_'�:o�=`��<I��i=� =iy��O�������}��}�=�>>=l䄼��=��_�\��=)�ƽ� �=��;�*���;��*=p�,���d=�~�=��ǽ�h=��-=�n�=�1�����쩻< ��9d��,�j=Ҡ�؈�<�%�= Pl<j�+�����X?�� X���M�=�(o� F9�O�=~!N�l=XϽ<�Ǳ<G�=H�_��_}=�S�=J�j��e��<&���؍��e=����-1c= cڻO����H\B�e���o�������Wi�׆�=\J=mF�E1��lk#�С�?�2=�f�mZ˽`�����=BƆ�`:9�j���Ľܱ��0=�bc����������vr;�q�T���Ͷ�ȿT��c���q��,���v/�܁ �@�뼾rE� -H;�0����-<�6= -�y/�����=�@��8=b<nZ�4�4=Ty�<�(���l��	���s�,�=P��<�5E=��[� ��:�:��R쀽�	;��0����<�H��\滼(�X<h:Q� 6���g�z� ��Y�;\��R4�p�y<"����%<��a��	��a<��� �����.���9�������<��Ҽ��[=����=$�w�8�;<n���=��z��?��J�;Г<.��=Pc���3=Z߽�R�<�Yü�μA�D=L��Z?#� *=�D����C;�1�æ/=�ŷ:bYv�6o,=0��;�l�<��'=�6�:�)���;�LP��3��B�= �y9�+<�t�;��;�,�<�On=P���B�<�Y开�=`tn�|���w�:@��̯Q=����PW�<o�H�6�PQ`�Ej=h�O<(n�=�3}�xt�<`������1�0�Z=J=���=�=6/&= <�~�O=-=�朽bE=�&E����<��{=�b>87=���;y*�=�KT=��h;�= �����j<h}���{=ڶ��A�����;�S=��콠2p�ة�<��< ���a=�Ï=֮���=< ���������=޼y=�󼊕�==�=����N꽠z�hgݼ;�	> d����<�~�=�M���IQ=,��A�>"�;�^�#�x~�=���= Ov<CE�=P�=�,��dϝ=�]=��|=��g���D�����;U�(�@�
=�a��w7=^�=<^rS�Г�$|�0;�2ɬ=n��ܕ�<э�=R/�ᬕ=  �
�;�e|=�5���<==X�=N;�>���5�C=�~C���=H�W�X�=p��<�����:;ം�󻼽��[��G��1��L�>�ۖ=�'x���ŽD����P�+jN=J�uѸ� 7�Nt!=g�� �r*��~ؽl|��&=��|��89��,:`��<@�<�[���������Ŋ�Z�<�Z���基��ּp�μD��l��<]�8��<�E�=��<K��4ܰ=zz)��p�<B�-�X��<�׊<����|�<pF��'V��U=�>n==h�p�I<p�%��񂽈�<̨ļv=�,�<��-���@<���`�~;@��Re����0u��f�m�,3�=᰼�{�<`6���;�
]=��5;�/�<���»4��$���'=0����l=�ʛ��#�=pd����~<��H�4��=|�܊��X<P=�_=|Ĉ� ŧ<����z�< ��`�4��ݫ= ��:&��L�U=Llb��w=��ʼ��=��'<?������<���<��<�@�<≼��Lo��63�jƽ��u=@�7��N=��R<�.M��s=2��=�T�<���<p���$T�<@>���ܽ�=�)��sP<������λ<�m����<�O���E�=�F�;[J�=D~%� �u;(-�~=����x�@��{=m�=�u�=�)�=P$���	�=`XX=�+d�� =R-����;f16='�d>�-=@�;�#�=H�Z=�C�n�P=p�?����� �Ỹי��6�=����B��p�|����:��t���40=\M�<���<~wi=d��=4e<8>P<��1��u��n��=��=P3ͼ'��=��=����}z��*I��`��>�*���o=���=$K�F�=Wy��LB>.������=}��=��<�.�=S?�=��V�K͂=Ҹ3=�+D=(�0<XAj�	N��&���j�q2<2�m� P�<�=~��Q�@{�;�z�T躼��=�;�� d�<���=�ա�8$�=XZ+� �(��D8=|��<���<��`=������;���=zL��q�<4T��p	�=
=�>N����<��9<����]��H�¼ P;e�=?(�=1��q�޽p�C��Ɇ���=�p߼�W����;�4=�R��X�E��Ay�s>��$�E�4��<r����� �<��<��=lܼH朼���<N�����;冋���� ���!�����J;=��;��=6�=Ƅ=Z,D��e�= ��t~�<�U��G�(����r� �|=�����w<�@�<�:~=`@ �hl0<��=��q�X�L�@p�<P֔<�P=ș�<ؐ�<�X<`᪼ ��:Ha�<���������5�:�|�>��$��B=Ҷ&���<��=�f�<C�=0�$��E�|���n�<���<��3=D×�6��=�	�<D��<�_�,��=`z~; }ݼ Z��$�=��v<�Ķ��f�<2����<@�<��V�2��=�_/=��N��=�H�(w@=&Lb����<�y�<�v� D�9|��< KA<�[	<�O����w�<ݒ��A�B+��*sg=D_�D�z=��!����pd�;��=�U=�ZL= X��@����},�gI��F�2=�4�;Ԡ�������U�̪�����<|s��==�
<YWZ=��7� L9���Tļ��ս��a��Щ<�b�=���=��m=@,x�ɐ�=*8=
���� =�,� �e��X�<QC<>�=h_޼�F�=�ٌ=x<K�t@�<�#���4/�`(5��E��U��=X/�g���5�ryG�*D��J#���<����ܽ�<O�==�=��= ��Das��Z��=�3=�$=mx��=0_=@�;c���6��q<��=�=�� �=���=�����=�dܽⰵ=�q����
B�=5��=��<�ݣ=�hr=X[���<<�<�d�<�q=N������ d+;�2ҽ�A;�򛽨�}���=@t6;.��t���*�n
 ���;:<�ռ�q�:X@{=�e� �=��L�P�;	`=@*�<�;<���< �9:�i�<G�U=���2�<2�p��<�ή<4���<�;<.-.����P�%����<�=(�"<E�3=	�����Ǻ0s�;��< ~�;�㯼��<J�<d�ۼ��L�����|���!���< ^M<to�<��<���<`�=������C M=Ў.��<��� ��<�H�|}���;P?0=��ݻM�<�$s=�/<^���r�=0�=����fD]�0�3� .񼰵Ǽ�~�=8x<ZA=�� =N�= <ּ��=��)=\���@�`��oc<T��<�P=����M�)=��< "Ļ 4���= �W<H&U<،>�H?<�FP�=�u<�P=�꼴k�<�D=��<.r(=t���	�� */:`&�; X&=��=Lڅ<�c�=+ =���<V�;0=���<��	���7=��0�V,=��<p��<���<�����);=h=��;�̣=@@�<@:=2�I�`��<�V�<�����s���3<t��<��v<'Ǽ8���+����������k<�S��ۓ�=��%�8�&� �߹}h�=��u<!.~=�)��"5�pEq�Z*e�`6=t��< �����%�X>��������;��(��6i<�|%<�~4=|������ϻ��e��ν�N�($��Z�=�pr=*"=P��K`=�m�< ʼ���< �X<`��P#�<8e�==��V��N<=��>=En����4x���%��Θ�%������=����� v9�t���4��r9�H5V<�=������<��-=�#=Ш;�F.��8�e��B���;�v���վ=0��<���<�Z8��)����=��=�.[�6=h�T=<���u�<a���@=\���>�#�R�=_��= G>;�==�^o<�"_��Ϫ:� �����J�=DI��xC��^��z�ѽ�� <�U���|?��2=d|�<T���ܱ8�XL��B�)�����Y� 1Y�d�>="d��h<��@��We<��r=j=��x�lt���"<p��<�b=��ȼ�P<�tż`0ڼ�X�;���;���<p�a<��< 9�9�<z�<x��<��"�*}�=��O;0%�;�6�:Ƀ;`z<яe= w�:�f���m�<���x96<dE�<01��@A ��= �<`n:;YR= G�:$1<X/<l�)=`��<�<�<=�<���<�
a�p�@���;�c�<���)<d:\=l֠� D
�h6�<�OJ=p��;TX��`�|�XJ��@L8���=��/=�s�<��<�~@<@p��<��<��O;�L�;P_�<8�&��0<�d6=R
��|K<tO)=45�<  9�ѻ�H&<���;���:�[�<���<,u�<nv=4	���9ٻ4�ܼ�ձ:�=c��,�<u�;p�1< s�9�=@8�:ժ>=,L	=��<�"N= �E�0 �<�6�:.<�@�M<�sK���9�$�=�^�=�AH<��<(�)�`�l<��==H�k<��O=�{�;���<HM��``�;���:$�!=Xи���K� �Z:�@�<�a���Ƙ��k��E����A�t]
�8sE�Y8�=&�#�[<Ԧ�<�M= %���y5=.:0��?�� m���ݼ
.=�v<������� ȼ��_��e�;�()��@.���;f�=`�R�$�,���<�B�Y�����%� ?���i�<~�x=�$=�4��G8a=���<P�|�U�$=���< �й� =<��=�R�<�^��;=W!<]��� �h���푼�]�������&	=`KǼ��ӽ��g�K����d���xG�<p?�H�c�(?<<��<�%= s������������8��H�<(�= ��;\��<����R���r=]h�=HI���=�~0=������_;"b��l�;<����b;�l��<W�~=�>t���<=��H�<�Y�� t�� Y����<�&����dԊ��*{�e�<r&.�&D1���;��<�깻4�>�D����z�l�������7<�=���<��<�����<�}=��(=%���`�r�<�N#=P<���XW�<^ļT~� s_;@bA; ݃<��*<�y=����ط�<@^�; �w���U��Í=@<�<P�n���� #�;Xآ=��?�p ��`�
=�k�HK\<���<���7���=����؟��@= �q�P��<�H<��=pU�<8Ƌ<���<ؾ�<�̼`~��Ɋ�0-T<�W���(�%�6=8�����	<�*=���<h&�P<������:�D��<a�J=h�6<dX�<�LԺ ڽ���<t=����<H��<x%p�Э�;�<9=D������x�	=�}5=� <@�ܼ�t�; �881�<-ZG=�=��H�_<q�<8]#��I�Ƚ#���3;�<8I�H�=h:[<�h��� n�rH=�pܼ��<(g�< m;sۆ=�H����<P-».�=���<\kq�X���&�=�_�=�%4���<숬��U��|�<�Z <��<��T����<@����N�俙�#�l=ع��
Ҽ��2=�p=X��<�����l� :�蕉�4/ܼ=@=���@R��<�;�aT�DϞ<����J�< �޻��$=$��<�5������=���=�'"� �;4�ټ�Ə����;�A�<�P��>��=Tˎ��=���� ���׃��=8�F=6-�=���=�����=���p��;r$m��.-<��A�P.g<�}=���=*0=�|�<��=]�=�ʓ=5=���;!���4=��u=���@��:pP�;Q}>=_)b=��`��s�C�<,�<0mü 9�<��/=(�V�@M����=`�üȫ�=��<�了 �z���3=�ᇽ��[�_=�q���A$=���<�#���.=dӋ�ꘂ=��̽���= ��;j�@/;;,�< �L���=A=�=Ȧ��&*=�<L=2�=��(�6<)Wi=@�\�`�;h�I=���c�=;`�=�Ǜ;�]e� /���<@�e<��=�y���<���=|��xc�<$%�<�y="�+=�<��f��=��=.�E�>t��&r��-�ýN�/=�(^�^[!=���ֽ3��0ʹ��D�W��X���Я���O�=d��<�n��� ��ڡ�V<=?�z=pd��y#��,�:�^�<�2��$ �<iܽ3X���u����=�r�0����1��hP%�C׼��۽*�(�T�9�f=��B�9��� 
��P&߼�.(��,0���R<ף����(�蕖<��߻��ｖ�>����������۵.=a�;�P���<�l<)�)=$�=4��<rp=b�w��T8=�F�<䁙�X�=~Ԋ���<%�ý�w�;x������  <  �<`�4����<��˽$!����x=ʥ#�y1�<K���P��b� =`VP��� �;vIr�(q�< ��.<0��=��ýI==��Ƚ�>�<�/k��4�8���pU���i<(�8T�=D�[���:<~>潍�H=��<���'��M$=�bZ�R�����=dr����ɺ�����p�=@x�:����T�=�U�<�y�=vg=?��Pr,�HIY<�<K��(��� �<M�<�ۼP��;�|��U�<D(�<��<�L�<��x�=  �5�{$� o�:���<Ig�=��d�8φ<f���4����☻Fb\= ���0�=X/;��]=^Y����l�����GC<�~=V�>�S�=:=x�=p5�< 8�<!�����N<a5� T�L=ѣ)>��(=%"=�R=�Ǡ=���<\��<@^��D����ϋ= �=fw=�_��ü���<d�&=���#=<̚<Z`9=47����`=�ҁ=��pX&��r=�k`���=��t= �n�g\<=9r�=�*l�i�˽ �λv�p�x��=@�r��ճ����=맽2P=�����>��⼢��I;k=��=���<��=�4�=
���΄=pLp=k�= p���Z��GH=�����z��=�2��=��=�j<<Ř��'�<�,!<`��;2/�=��/��`�<%�=P���1�= %���&�<�3E=H��܁�=��=҅��z���iۼ�x����=6�5��]K=��<U���Vdy�P�����J󽐝t�@J�;��>��=nY3�Rx�XK:���o=|��="�P��i'��3�;q=Z>ƽl=�˽c ��LX��< �=$'/���r��c��@Z�P��Yʽ��x����
���c&���t<�����:�(�g�?W&=�/m�<�<�P=t��<]��dL>���P���A7ͽ0�<0f�;φ��L���I�d9=~i4=d�Q=�=���`�c=ȶ�<.락��F=h0m��
�<��S����<l�"�vh��h<J=`�K� ��;f?Ž��Q�0��=�� ���0��v�� �=��u<��;�Y�26k����;�e�<��~<�K�=Z���L�=N$��<5���P��<�:7�\�����<`�;:��=�8�<C������/= �ܼT����4�=8������d��=�[��	�<�/>�w�=��5<p�]�y=�s=H#�=�=L(��0&X�@��;v�H�:�ٽ�W= R�9 �J;��5<������ =i�M=U=�= )����V< (������ =0{�;�\=Dj��D<�Q��8�P<$.��tL�= =��em=ė&��UG=�hh�h��;ۻ�x�M�pU�=׹>	��=���=4��<,�i=�%=nL� ��<2��p���9qG=P�=>��3=�c6="�\=I�=0Y��=�����޼*�U=w�<���=0#Y��KW�Hz"<P��;8O��;�<\�<l@=��;���=�R�=����0���`�D;xqw�F��=ف�=��¼��=H��=�O1<�u���x-�����=��W�<n��=M����A=Z.���>v���}�G$�=c$�=l��<��=j�=�N5��*�=L�=���=|��<Ķq����<�=׻@�W��J�<f�\�k0�=�j�=@�i;�U��&-=(wD���W��}�=���Lվ<���=@A/�;��=��3����<��U=`��;&k�=Ae�=��5�t���<^[d���<�p6���b=�U=摽~� ���­��L���c��&�<Y�>�Q�=ܢ��x�
�ܜ$��{H=���=����d� ��<R�/=ǽ��<��������4i�4(�=�̼`�}���V<m��P�^<���T�ݼ��;�뽀�:���h�P<�U���hA��4<>�O=䆬��=)ɋ=��O=����}�=v��_������Ӽ��ټ����P�<���)Q=�=<��=� �;��;�Cp=`]5;�Ry��YR=���&=L���ߒ]=0���j���=��K�h=йr� GB;�$ʽJH�|<>lܕ���?<���;��; ��=�4=�<�֩�%x�� �s;���<��=j��=(���x�=D���T�<�f���G=�Px�8K����<�.=u=�Y7�0FּĲ���<�E��Hpj�d�= ��; ǐ��;�=�<�;=V)
��-;=XtU<9���]=��	=�|�=he�<�k���������;�l-�t#���xw=�������<X��4k޼��<�f�= �K=IĊ=�aM�p (�蚴�����1@=x�5< %T���y��r0<����F<0w��{�= �;�h=��N�j�=@�)���M��h½��t�<gQ>2[t=_d=�F�<P{f=�U�<b�L��l�<��Ż�K�T=$c5>Y�`=�5�;���=��=ؒ漀�P;P�D��{3� �:2�.���=o����lA�<��2������О:�*�:01�; �2:\�t=޽�=p�;95�4%��Ҕ��<]=˒D=<�꼦��=�f�=b�<W�����(OL����=�F�� N=��=���!<S=���m�=2��������=��=��<b2�=@�=p ޼�c!=(�<�+=��T=�x��,K���<���� D<�ϥ�<J�<r��=P|`< ]�`s�;���������=��Q���NpM=Ll�T�-=�р����<�9�=�y<^j=Z�m= �; �e�Hz}<­M�'e<�8\� �X<�g;<�P��\�p���G��d�@���!.=�S�=�$�<0\
=S�� �n��5=��/=�4(�B����<�=��p�@TF<�>.������U,��U=�V<���<��<@�+�8��<��'������G=x�ý���<>�M��=P�k��X��u<�9=h�м���<>�a=��<�酽"#�=�,�;(oX��m��t�<�X"5�*�
�0t�<�l�ȩ�=c�)=�g3=(Tʼ0��<�h=���< �μ0W�<��B;P�4=:{	�
4�=�� �%�tK���i=�9<`,�;�t���y��x� >�W<���<�P<�-�;i@�=�h2=��
= ޫ��H'�`�y<�]�<�B@=��=�I�<G2�= ~h;�r�<��ͼ�՗;0�u<�x˼�N<�5_<H;���f�������H�I< Ć9 a��e�`=��= #P�6��=�9=hM=����0=��;��s����;���;R��=,�<F=���ּ�\B��~�8�b���<����C*=F�)��}�8�6���=h�=8��=�.��Wi��B�w���]�)=(6�<�E���#�@X�������;����!=�B�<��W=����0<�nż�� ;��Ƚ�s���rc����=pG=�w2=�U�<s�D= oO:� ����<p�s<𝔻Ln�<�D>Cnm=\��`HN="�=�>D�,ȋ��Kb� IG�p7�'���&��=�ɋ�����,<�V���6�DA���n?��D�d���&;=�\y=��<l�L�l]�dk��P��;0hz<<`ϼ���=�"?=fU=��e�l؅���O<�&�=�SO�p4@=T�=�� =��S�F=8䜼�����=���=臌<Ƥ=�{!=���:x�1<@k�;�;|<��}=�[½h*�� ��:J����O<?������ʙ=Y	=0��;$��p#<2�=���<��d�8_[���=�`5�X��<�1��$?�<zڍ=���< ��:PC���E<@��;��<�K���;����,�0)��8q^� ��9@��p-<Z�� R���5�<�J=J�:�$Va=�����U;�X�<P�w<`C!;�?�<�����j�;�ӈ;8� �1�`�1<L#��`Kd;�|=@��<�4ջh&�< �G�0��� ,���=�MҼ���;@*���)=0;0�4���� ;�x�<�-ּȤ<�(2=8�H��|����=�0=�::::,�dӸ������-��x<$9�<[=�Ҏ<�>,<T����S�<\P�<po�<�ۊ<PW� ��� !=��޼���<؋�<�=-<H�`-�;h��<��+�L����=0<�G=�s�< �=`���R��@��PM�;h�= �k�(7<< ag��K<ß;��=H�?<ϻ:=��<��<���<�����<`u̻�Qļ��9�\�O����L4�<H�=��2�h�<�rܼ�ǽ<pN*=� <���=��<̽=8�M���u<l/����3=����=��+�/=8��<�)���μ�'ڼR�ʽ`�뼀�������=r�I���߻ ';Nkl=<�<�rC=t����
������vY�j�=D��<(Z{��VS�p/w��׬�h�<�V��8m<�7<��$=ic�H�� &J��k<𭽤iѼ�����&k=��t=*U=8�~<hCK=��L<��Ҽ�"=�w�<��!�P
=.w�=Ap=� R�$:"=��b=�~D��@�L�� 0޼�@l�ۄ���y=8T[�����T����������@�:<�;�����<b�+=Q�=|'���q���3�xu�Pw�;@���=���<T{=$
Ｐ����#<=�Z�=��f�5�9=Ulb=�R��0��<񼪽���<`e4�,���O=��=@�;.`A=�;��{< ��90�'�pkq<F;=Ћ���̽0{b�����m�<�(���\��1�<���<@u<8[��@���� ���2���<!�g=����`��<�iY�p<�&�=�X=P��,ID��]<0W�< T�<&����V<8¼ %`��=@� �����< 뜻��<(�����!<�4-<@��;uh��La= W,��,��Paʻ ��:�����=��Y���D�Є�<�����;8.�<�fj�A����#=�n<�X3��j=`d�����;��4;[�=�|< s;���<RC=�}���B"� �z��x�;X	ͼ��:�V=��¼뀻i<~]=��k<(%q��QQ�h ���0��X�p<�D=P�;@���І޻�˰� c�: �[�p��;��<p�|���;��1=���$>���L=�='=@�G;�)�8K�< �?��L4<4[=����Ke<
�=\g��
�����ǉ��f�<8랼�g=��4��n); 恼H<|=�Z��<�l�<$>�<�J=�Ҽ`C-<��;�b3�`���H���(��� �4=#��=`Nջa�<r$%�0w�;"5=`Ӡ;�\:= ����9�<��� 4������ŋ=�����ͼ�f=�;s=`���k�� �!��%��8���{�p�=`���`�;������ɻ0��;@����s��l<�`=�D=�J��L��J\
=��=�Pϼ�=�Ѽ����8�h<��<��3��Ff=�1L<�P=����,��<<���ق=S�=id�=i��=�?�O^e=��L�𷤻�)s��(�� ��#<���;��@=j�p=�K=��~;R=	=t�=�q(=p �;�����=�1�=�����<�8�< =4�=pK�����<���<�=��@;�2><`��<��)�`V�;�W�= i;|�A=���<��'<����M,=K���zO.��Ҡ=+U���ǰ;p�J<"�r���<�+���,%=ͩ����=��;�a��X�%���X� 1ϻ��=���=�׋�`�D;ore=u��=��1�$��<��= d7��>=�k=h�i�M �=�l�=�'ѻ@x�:pK�;4$�<��<���=���`��;D�=8�< r����=�c)= ��;LA �$��=l��=�e�d�$�ݎ˽;�R=�0��ؒ�< �<Q}��0M�,����O������X@N<>��=��
����x��`�U��==���=��e��4�����2�<H�M���<��޽|l�ܱ�����=(�ʼϠ��G���m�2��凮��a���߼h�н^�����[�<kQ�f2��pZ�0u'<*/~�ȕ�x�\k���ؽvx<>�T+��C������B= }�;�N���^��D!��e=l�=��r;h�</Z��F�=�Yb=�뗽�$=:Ԑ����<���@R�:�⑽ ��(=$�<�2)=��I<Z�Ƚ��J�c�=D�Ƽꀼ�������u#=芚��q[;�A�<�u��U&=@pͼDD�<#ɶ=������<�������<� f�����E��<��<8f<<x_c��;z=�}���\��4ə����=�d��z��1�<�M�v�C�z>�=�O�< .�: y<��l=v��^Ol=��|=`&һ�v�=2�l=�V��D�,���;<�+*��D��Xŧ��;=�*��VP< Ѹ�p��<�E��D��<��t�0ږ<�-=�$�<<p"��Ӽpp�<�`�=����6D=��������Ģ<���<�㼿r=<ۍ�]�=h�����<Ɛ.��F=
=v=,��=�ّ=|[�<�Ո=3�� 5<�6��X����^�`a�@G�<��=�(n=�S�=��I<�.�=��7=0ݝ<L�<��Լ���="�=��ͼ@�ռ�r�<أ�<Z��=8��<��R= 7<p��= �u:x�=�=������s;l`g=<Oʼk�=-.&=`��; ��؀�=�K��3㰽H2p=q����k�< ��R��0��=L%��tR�<a�����=�����-��Xp�<\JM=p�<�C�= ��=�^m�"�=h �=���= !�; ��2z�=����$��<��=��¼���=1�=��; .�9�#1=X��<�i�<z��=2d1��=��1=F�=�'=P ���,=x=v<�٫�=c>�'�=|x��|���,��%�½Θ>='���F=���<���~ �x�4���j7�ބ��F.=6(�=DG=��H(��sͼ���=�Z�=��k��<.��7�;�<*픽�l=3��V�9����#Y�=L�޼>D������;�\U���۲�~Q��;����f:&��2��=';��!�X�Ǽ\��<�wM��KF����;p��;Mܽ�E@>
�""�*/#���2=�<Sܼ����P��1s=j�A=��=BZ=^�[�@Ӽ=Lt~=v8���˅=4���[=u��H��<����r��PU=�G=�x+=�z�;�׽�U�����= �4� 2��`�<˺�����=p<�0< ?�<��r�D��< �ͺ�G�<p�=�H�ƁO=T���o=AQ�����T�_���;,6�<HI�<��=�jp�*�A�P�ֽ��=@9P��=F���=�����i���=��O= � =d�<��v=�y����<;Qu=� :�% >dID=�"ȼ �żXÍ<� H���ս<��<(Y2<���N<d5���-�<p�*<�9=�E�<@�<��<���;<o&�P@)<�c�<�{�=�����B=彀�;P$�<��K=����hj?=`��e��=������=��x���<I�=�K	>�؃=�DP=f�g=��H<��<kה����;>��0����@=���=���=�o�=2=�� ><��<�G/<��<��'�\��=�J�=��=ȣ,���8;���<�H=�w<4�J=�)���X�=p��;�^=��t=�����Ȟ��= �4����=�IR= @��D&=�T�=(�<󖽽t[�<�g�D�= :���H�Q�=:�˽p��<���Ю
>X�����μ�r=�Ӫ=�=��>��=�t'��==j��=�O�=��=������=8o��o�T�=�50�y��=Gw�=��; <W<�W�=(�<L�;��>*R���<�?R=@�<��u=Dy,�b�8=D�)=D\��l�=)?�=��(��Y����ؼY����=_��:=��#=��x���ؽpL�vH��h���4��e�=t��=|{y=���v.�X�3��=�y�=�
R�� ��V�<���<tA���� =������I�\b���-�=Xb:�0!���
�xf��U��=��"�#�`���{�t���m�7�+=@�b��_1����;�*=|��`i;�6�<�m=����}>�м�D���f�h)<��Իo���@��p~<��6�=��?=��c=�.�<�Ø�L�=�L=jN����=H$b���E=�a���==�ǽ��~� L�<?�K=�4�<0�����ֽ���f�>��%�h0��I*=.*6�65�=w�-=�HC<��<&}s�K�<��_<q�=��=P��h~�=8�25=�U��`Ω��B��d���0�< ���s0l=�F�a⋽�q׽BT=T_��'�,��=�l����5����=���=�f=��;�,C=�l��-)���2=�F{:�>`��<th��T��(��<��I�ݽf�C=x��y���nJ��}����;BP>=|!=59r="�<��f��k���%���=<ث<XD�<hm9�?(=�Ⱥa<�#=m�^=�嗢r�R=�7�'��=XEq�F�=烚� Z:D�<qh >�@5='�j=�BR=@Q�<��w;��� ��<����0�^=�e>�	�=�)c=��b=0�,>฿;�9���9ﻸ@H���=��s����=�������`U(=P��&:'���<$����<��M�I�v=rI�="�W�|]� P��́�siQ=�=0zD��G�= �=$޺<�Q��p_�����w=��i��l�<���=��(�2=>3�>��=�A���.�F9�=h@�=<�=�[>��=xؼ�=��l=�`=�ʆ=dҝ��l�= ����ڽ�i�<)J���Lq=^�=0X�<�g=l�Y=�=|düd�=1���@�:S6=Ϛ���0=K��r);=�4�=4ٲ����='�=c�;8`5�(��� �����<_��X8E���<�jI��'��Lk�l���ͽܽ�y"�=�Ҹ=���< �";���8�o�x=N'�=���5˽�L�<���<�6r���<����$��T@��f�=�$�<�x�:�RZ������d�ʕ9�
���e_<��
�dE�6���f=@ K��TT���<xV=�����;���<<��<��z�D��=(�
<�2�W&�O�����jLp�`�];�����=�8=�7=��G����;�C�=Ł^=�&���'=�0꼮�,=�!��xf="i����9����P�d=�=�[�([ý a����=p`y<��S�Z�#=l�ټV�=��O=x7�<�yO<Q&� �}<���<�E=��=���<��=�g3��]A=��O� �ļ�!�; �ļ���<4	Ǽ��[<��R��;r�ܮ+��l�<|�Լh_Ӽ�f�=`jB;�l߼͍�=���=f�=<��3=�r��(Q<|˘<�ۃ�	_	>p �<jx�Te-� �9�1�V����=
��� uX;>Q
�j-�Pԏ����=��=���= �7�PiƼ���^��H02=�٭< �i��,��G�<��#��^�<�� =��5=��Z;0D&=���]�y=���`H�<-����I�X���A�>=�<_=�[=�>�<x����>���<`p<�;׻ ��<W� >�l�=I^<�0[=4>p������x���@YI��C<��9���= ������<2�o�ᰬ����;H�Ҽ�o<xż�AZ=;>y=������������'��P�,<�f<�]߼� �=�D�=+==%���@[���7��V=h0*��9�<ۭ�=l��=$(2�(L�=�໴���팻=~'�=�y=3'�=xM= �Y��<6�	=�c�<��=�Oֽ`=�����������<ύ����<S��=�,=i�,=dO=f�1=�*U���=O}½��:�(|�<�M���<E��6-=��=��L�8�B<��<�?!<�C�`�I<^�$����;j�/��'n���=�(Ο�V��������W<�惽8Hi���r=�oR=O��Ƣ<ztn�P�����%=��=��=�hO�@Ѡ� b���K��A�:�_� �9��Ķ<�=@�><����8�[<��w���ܼX�����<��T���>��;�=`��@Jʼ���<��~<�$� �8;�}�<`,;\��==��3=x���w����0'�v���; ����(= }X<x�H<=��Wg<�A-=��=�l�;��j�Ł�y=V�'��J�<4�� `�8��3���<�=4b���l#���o��p�=�{�<h>�<�d�<��׼TM
=;�<q(=`)�; �<`�����<(�<�7�=8�=;= %o<��Y=`����&#���g<���0ûl���B�h��2���T-=�|<����T��}�&=�, =��-ĺ=l3a=�K=�� <�`�<3漰�V=�(�"y�gV�=0�<:8�ĩ��o׼F@	���s�0�<����m#=FmI���l���qx=��=s0!=䖥��`���v�7����=hdW<�"�t� ;a���=�� <z�< �9�p�<�}/��t�<x��va=������Z�x�Ƽ>�=�eE=�X�=�-=(�< ����]�ھ<X��<�'+� ��<���=�_�=$s����0=���=�:6�h���F�l�4����*�<�=�s�
C���:��鈽_:Խ���Я��̹�<�iڼ�7=)�r=��C<Tl���8���d� u��]�;4�����=�z4=F�"=V> �)�hn�<0(W=��l��2=���=�h�l	�<l8��J=@�߻����8�=f�= L�<���=�*�<��n<�'���<P#4<��~=�½hTK��ѼX�Խ|�<[���\�<�&O=��< =��=���<R�V���<sT���;�;���<�D��4	�<�ן��=`<5i=@o�;�]-�4\�`k�;@鸺lb�<3
�@K�:����*j���܅���*�ુ<�g��H�<����4s;H��<B�<y:���f�<��Ƽ���L<`(z;�qg���I=T$��,���.<�ؼ�{#���\<hE}�8�Լ�_$=��<�n��{A= �d:��ڻ ��`��<����ޫ��W�<ȅ0=������� �8:�X�;̔�� g�:80�<�oٻ �ٻp0O<Vf=p(໌׼P;���A񻾗��q'<XyQ<�*e;$ڢ��U��&��p�<l�<�J�T��<8�=�`	;J! =`���+м�t_<�}= ���k��E�<.ἀ������;{�<�EW<�{=�Vu��IۼP�ٻ��y�]�= ���p�$=��Ӽ�"�<�	A�2Y�=���<�<I<^E=�=�Ƴ<Pk케_:<��i<;�Ƚ��h\w����@ҥ�i�=�.;�]^<n,#�(E�<Fy7=�8p;:3=���<ȅ�<�d��ӹ�(�n�=�^��ռ�=I= �^=F3!������j�����%(�r����I�=�	 ��@�;`�[���2�`�A� �;"�!���<
��=���=��ѽ.E��̪�<2��=�ۼ�f=p��a�����<�h��w,���=u�&=�B=�ǜ��4�< ؉;LŔ=x�<�x<��=�L�,�x=hX���"��B�e�ĢǼs���ؙo<�n꼠9<�c�=��<������;��T=�%<=�>�:�>׼�y�=�>4Ƚ�<<�<���;z	�=䉫���=(,�<4��<�4�<��k��h�;��p� -={N=1�<�[< �=;t��<L3����<g-��Lg�5��=����(-y��`��K��8xټ��o����<	Ȝ�i�=��I�x����y�N�8��Ll��NW=%+�="}���-�FO=UVI=���:wl)=`�g=(���!��=TQ�<Ё
�&Ŗ=;Fg=𐕼8�r<��G;H�<n�>=&��=���� �����<ѿG=^ ��!=h|�<T߯�ķ1�x=p#5<n����X�z�{񉽤Ej=�M(��x<�!<7̼]J���a�@:�;���j���-<��=N�0�x�߼�uӽ�Dc���}��V�=��l���dY��-�;,��<D��n�g�h�V<�w?����<���쾄�xq��x_�<��*���	�>�U���6:\�o�:zD��]*�̍�<0B<�p�C8��0︼�
��}S��0�2��,����=���;�uZ��:���V=��#={�½ �g��k���<���<������6<��R��=(��<~�������D*��<<$ ����썕�@���(19=�t�S|=P,�`�,��P9=��< �f� aX;�<�2���]�<",O� <Р�<&-�����L�0��"�	'k=T�5��ű��Wü���<����0���x���H!=L[���� � �<pQ��pU�;L"ͼ�a�=�
ܼ)q��85q��>ֻgh�����<LΧ<px"<P�{<���;d�c�8;�=���<8��\x�=y�@=��*�TD� �<ah	�07x�T�e���`=�ܵ�<@/�;@��:켢��n<�:���<��V=y�A=\��檌�8�q<���=�KżU��=0l��
D���<p ���r�Zr5=`�;_��=�]���z7=8�ހ�=��&=�0K=�ă=H����=��w���I�Aݢ����]X��@��:�	����5=b��=��=ࣙ�u�=��f=���<䍬<l<��g>���="r��`�wU=ۡ<~��=�V�<�?=��; [W=d��<�SL<pe�;[�ѽ�0=�d_=@;���J1=P\<��<f1���x=�����喽$��=u���hm�����;�뀽��W<������<-Sͽ�9�=8=�������<�p�,��<�D�=`�=����VɻAA�=hj=��;�i<���=8�����f=�=��i��=�݊=D���e�:�k=3�<��m=�ӹ=�}	����<h�_<ك�=ȋ<�wK;"�=��[���J��k=��!=B�)��T� ��E���1f=��E�h,�<0�<0`���$཰o3��!7�_䶽�*[�G=J�o= �`��x�do�pMӼ�B�Ԩ�=&A�*.n�����`qk; ,T9p¼�3s� �;��~��2T=�>0��������H�U<6:�N�#�6.Y� 5Ѻ�
���ts�����B+=P.C<h���_}�(�O��eۼzG��:?�4bּ"��Nq>�*�;���z�"�,�V=�2=q�۽@�;L3�`:(<P�<�V��8/�<G���诵=,{�<̪����<$4j��� =I���ހ��w׽��O��5W=07
�H!�=N%���[��=�S=�p�p�T8���0��ȋ�=.�!�@�#<8�<��3������� j����_=0��xOR�x缐#�<9��pj�����6=p7λ�	�>�.=�&�� �ļ��o�|ј=���˂��0_�;$������C.=dd&=���<g�<p��;G~��7�=0`<`>{��>�X"=�g'�l��b=*�9� �ͽ�;T��<��+����<�{��_<t���(�:<NI����<(ot<h��<��)���:�hYf<-�=\�ּ�®=��� J���89= ��9��P��9K=�C��"M�=�>��m�=�3
�/0P=j�=��=��==�6�<D�=(�G��
��v���q���B�Hl� �b;3�=���=䍬=�bM<�6>zu�=x�;�
=��=��B,>�ߦ=�<�P�g�CF=��(=���=Pk�<l6=h�Ҽ�	d=�e�<�C�<`��<�W��@�*<��6= j�CSQ=���<b�=@��;׆�=Nf9�o�ν���=&�`�,ɼ�u�<�j/�-�j=��� ��<���|�=$���WŻ��L=u�"=l�.=��=�^�=T�X�4��<���=>h�=��<(*M�h�>m	���;pEL=�bۼ���=�1�=�A�����<���=��<�eD=���=rb��(�<���<��q=Lu=��&�$�:=xT�<X d�V��=Uqq=��ܼ�6h�@�����;L=����u�<�=���� �`,c���ʼ�r���fC�g��=�-w=�$�;��]��g,�*㼀<[�:��=^Ek�9�����`�n; ܦ�K�bx� ��:�A���=p��x�y�*Q���:b4 ��f�b�n� ��:V����[�������E=0\�;�7ɼ�$� ��;�����O��e(���
��F�dT> ��< �A9��6���=$��<yrս|q�<������<ڸ@=H�P��,�;6oi����=4�=Cɽn�%=�{I��XB=Ֆݽn1����8u�ο=���s�=��=�V�|��<��= i1;�h�� ����hɽ���=�J�X�<�f�<&�=��@�j�� _<lÆ= �:8;<<�!��8=���pн���H�T�< Z)� ����1=�i��ZZK��4i���=h�ۼ�T��<��ƻ����"2�=>��=��R=M�<��B<t=t��Ӻ=��;,-����5>�$$=������|9%=��P�������<��C�z��� <��ļ�Ju:P�H<�@4<�f;�O�<`Ѯ��O��4����4�<-�q=�Q�HY�=iu/�pt=<��f=�4<�1�ks1=p�1�ڎ>����6��=&{V�l��<��;p�>Z=�vi=R�=�����1���쏽@�a<`s4��=�����<�F�=��>��=L^=�W>-�G=('� �<@�1�r�>�D�<�4=�^^�����P8s=��m= �\�p��<6N��j@=�G<d�=�;=���,�Фu<�`i��=��<�ɾ<�{=���=�����0ѽ�]�<��
��ʧ���<�eػ�z�=�3"�܅�<D�;�b�=;���"1���=
?�=DP=��>���=p����T�<�-�=�Dz=��q=��]�$�=X���>�J=z�h��ɰ=R�=p�Y<Ђ-=��=G0=��k<�7>/z�����: �C<ܸ�<,�=�宽�uL=�я=�NX��"�=�r=�e��D)p�h�ͼ=s���=	��� {U��U�<Hh6�&��� a��S ��$��`K�k��=�T= q��@�f���H���c�<lY�=����B�زU��9߻8�/�0!1<�g���X<\T\�E�n=���;G�f� ������	��8t�x�s<�
��J��kb����=�Y<�� "�`����<�u���H��6�H=��:�$���>=hV��,�i7<p�T<i�����=�9k��b=�=s=�_Ҽ,�ټ|�Ǽ���=�_=�@���/�<���{.=���dtż�X��p-=�@�<�PX��)�==���P��<	2�=�r�< |;�g<oթ��ş=�K/:���< Cu<`e��XCp��cp;@�~<�~�=��$=@��<��ݻ��|=Z�"�(KP��ؼ�J�< ��:� ���<��ż6 J�0O�;hz=8i���(��x<d�<��u��
�=��=��=T��<�]�<xC-� ��= /5���X5>==*�,��`J��	<�?�,���p=�s|��S���z��V�]�ӂ��� =��h<�0�<@!�;�S�����!
����<���<<��<,ܼ�)I=��9�%&E=�M=(�7<d�����;8k���=�*�1�|=r6� ׸���!>�S=VN�=%s�=��Z�p,��r]1��Ę<�S<�s��հ<	��=0��=Jcr=�-7=r�M>�T��h�_��L;<���=H������=�]���vz���&=������<��|3=�a ��:=�OB=-o������d���o�`]�;��9<���:�P�=���=�{�<Ap������q����+; J���'�;J��=�t"���<��@�,�x=@a�:`o��>�=���=#�b=#��=D,L=@��� ��;�r[=hu=Κ�=z����@=�(������ =I���,�b=
��=<m�<��x=�O�=��d=��L?�=�� Y[� ��; �j982�< �νGG=4�=�d��O�<�~�<�&2<�i6�@�K������^<�sp��Ii���`��;+�(�=����<��?�h��r�=�	=nc|���<館�аG<���<4��=�>#��: <H/ϼL���|U<œ<l�pn�<���� =k�@= ��� �ݼp�?< �N�h�q��4)����< ����6��޺⣴= �z;������;��;zLf�@�1�О�T7˼>m���=b-�=����{���p�� ?�o�����,=�?<,;i=�=��C��7�`X�;��=6V-=���� 6�9��+��#=��ʽH=ͼ��H1< ��x���-�=jMG�J��<��=�'=:�=0 ��M����<�D:�Z�I=�< ��<�O����<�y/<V�>AH�=��<Ё<d�=�� ���� ����!=x��$��h��X<ּFl�8��=\W_=P_ۻ�$ ����;(k=pTʼtU�=r)�=z9b=�8<l��<T�ż�5�=$Bͼ�Q���>@��<�Q�І!�x���[�j|ǽ� C=h�J�P�C<R�����d��v@=D��< Dǹ�����5��z��Ɯ�== �؈Y�����r�<ߜ!�p�=��=H)<����� Iѻ-\�=�߫�6�_=���܉������=t%=�{�=�Ō=І�8vF�HV����i<���<4*��`�;y�=d�=��
=<,=J>�V#��G���d�������<T4׼p �=�j�J��� �z�3[��ψ� �[�P���Q5=lѢ��[�<J�o=:�0���Wټdo\���-� �:�7"���"=Z<b= s�<�hK��V��0?�;t@�<x]���N<p�v=%���:�:+)��=0��;У)��A�=(��=�yZ=�}�=*�=��^<�]w�ٮA=�*�;��=l�Խ�zX��+z�^��X��<=w��8�.=j�=l&�<��\=~��=#8=��x�&Q=�^� ��;��!�����U�;�Yͽ�ރ<�� =�ə��!�P�ٔ<@]��f���?���"�;Ԓ���F��4����<���@�Ѽ4}�<茻��w׼�M�=Xhw<�����4=�Q�0l�<P>Q<�[�<�\�YH=����<��k�<@+�H�-�8=������B�V�=��.<Xt�(U=8��< �#�thɼtC(=<�<�����F=��v=8� <h���`"�; q�:�x���	� T���aμ��� e=���=N�=��͜��G��X�����G��W?=�<�b=��:��f�&<[�jv<���=�;���<h�b� ��<��=�����+�0�;�.�< ����~�_��=r0�"��xq~<i�=���<0�q=�M��j�l�@RV���
�5�Y=�����,=��%��;<�<8�	>;�=��<0�=�؊=��<����`u�s�N=pq˼��\��||�T�ԼP-���=gHG=��;��=��YĻ��= �}:���=R��=D��<@B����<@tF���=|5�:W	�٣-= HT=�� ?W���/��2����#:P���h�=��*<��Q<�t ��7��X^r� p��3���0e.<z=���=���� hT��^�=�����=@z��fR����<@��]腽�}0<�[U=�J>=t�k��z
=(�<iV�= �}<���n=�ㄽւ=�ވ�еy��i�f�4����o�<�v�`?����=��<V	�����M&=�P= ��8�����p�=J�>�̿�H_�<�T�<�f�\��=8m'���0=`�G<0��;(�<pa��l$�f�V�dF=��=Ho=����������<�h�� �(<oI���R���=����pr��D0��ӿ���F�D3,���p�mS���.=�|����4��}�F�k�P�a��@=��=��ؓy��/!=h$n< `+���=�/=��Q����=��';������L=T'1=x⩼�D?=��K� �=�6]="��=8�D��/�`�;[��=.�E��(/= �:@����@T�|5����/��ü03�;,����e���C=�?�0��;�MȻp�;�#X������=�; <��M��\	�p<<�Q���i��"uj��P���_��J�<��&�>�e=\�r� a��7!e=6d�����h�,=H�üȆ(�܌����&�@���!:=�Fջ�����4�k�<دd��.9�<��<�_q<N�< |;�؛�F�*/%� 2V��vZ�V|c�  e���]=�=��c�q����=�w$=�������<�K���׃� U�;��|���������a=`{���0��x���E�\��<�;Խ�ڽ�y�T"�<ׂ=�U�l�=��.�@�T=Vօ=��z��B,�k[/=�؀��ڽxFq�1���X�~<��غ�����=����|ļ��b%=�O��H\��;@m!<8�7<𤼌�q��}W=�Ｘq�����А�;��=��=��d= M?��c��l�B�Ht�<��q�೾� A~:��x;0��;�����YQ���>�����G���x=��=*~x�̲����<,�ս��!�l�i��A=�)��P��<H�<�ȋ�PN��@4�Ӭ���
<>�0=H�d=y�Ͻ����0��T��=����q�=�&\�A]���<t���/ ��H�<hs�<8ϯ=�f��y%L=H�<��=P�< �߻�xJ=0\¼��=\���toļ�����������`4Z;냽��=F��=�Q�<�m(��_=��=$�'=��p<XB��'��=��=�̬��м B9=���;%�> �h;���< �u9pCD<-�)=h�:���׻S0Ƚ�b�=n�-= X�9�y�<PK���6= ��N�<�y��ɒ��j�=sH��T�����=<���D��K��حN<��߽��v=�	��t;Tp�<fw����<l��=�X�=�O��P�2��UP=�g�<�9��&�<���=����=��<@k����=CA|=����p�E<�i<`��< �=I�=�^����)�(RH��=4����u�;pzO<�Ɨ��������������NS�@���Cx��W=p�@��<�I�pΈ;����HѼ�67�Xr����t���;|��<V%p��m"�ת½0q����W��Z=�Q��m�<�\��h;�85=�U4��П�
=�����<�� �\�I�ȌR� |�<@뚻�ʆ��`m�೗<�4�
<U��'7<��<���<��L���D�ż�@�;}�fы�zd���Ļ��=�+= b5�����(=�=�穽,T�<p�޼`�3�(ʈ<s��������5��VJ�= bF��i��`���S�tf�<7H轏�ѽ�l}��d<��=le��ơ=�8��	=�=(�)<He�B�=
�o�1^佐N<�ّ�p�~<�Cٺ��Hn�%*�p���w&=H�-�x輸h'���<����@���F����N=DC�`;� f�; �+;�q<�,�;m��=���%E���R3� »<�.����<(��<�>(<@Z�;���pv��
>t���ec�׼�=H��<�H��0"�Z�X=P�.��`������*��$��t��<���;f��xXҼ�"�����X�< #��z�<�P���� (����=X������=����:��l=NY+��w���t=�J��v� >����*�=yżh�= �����)=ྠ<�1W<?2�= �ݽ�������L潼����䊼t�%�+!�= �>,o�<��;��>r��=P��<*�=�]���&>�=TD;��k���_=�Z0=E�'>P��;�rZ<�n(���	<l1=p���@�I;��(�G=+<=dS'��n�< U�:��_=�_d�d�<���]꽬�k=��'���C��~i=�G�����.��e�<fe/����=�a���D<Nh�=�hl;;�_=�m�=ʍ�=NƦ�����B}D=��=p8��8����&>T�L����<Q)=|ϼ���=k�=�2���Ih<�e&=�=�h�=�1�=��I���v� r̼y��=�� <8�S�\��<D�<�-޽<��<@K�;d���ķ��A�����V=�6��y�<��: �2�����&��ܙ������ǈ�<+=��0<*�n���>���σ;�	-�V�=��Y�Pϻ��1� ��:,�=�޴�r���q=�L�H�<K��8?Q�(Y̼ �< \R�Tm漕������<,��"�X��<}	=��u<�������(�$�����5���Ý�΁m�����Yn�=�k=P�?�'���<4·<�S��(�=d��(B<<@$=�Ū�,ǟ�ʒ�1b�=��U<�f����<�C��$=<���*��^ώ�@��:�G�<�AW��R�=�M�@���t�=��C=`�Z��@�<��l�s����<�6�� T�<���nM%��I���1�XlV<�K�=P�<��c��s�d�<ثؼ�����k�K=d���X�E� �R<���`�;�6_<��=@������tTB��� =	i���q>=��{=`�<�xX:�m�;�,{��>@���X���]7> =W	�������=��N�a
��<R����u!����<�����J��$l�X^��Z�h�@��<J�.� �:��&��a�p�J<߹=�Ѷ����=��4�`�V�a��=jq:�״��j�f=��d��<>Hp�	H�=��-���E=����i�=���;KZ=w��=0����&
�7줽���b�N_�@S��<;�=�
!>�*5=�om<�f>?��=\��C�E=�G~��4>��<@<ر���@=�z�=.��=`<m�<c��� ��<P�<�pE<�PY<�n������<�;���f<�O�<yC=��=�d=�ڼ���꽐FI=����8I�p��=�͇��5=��1��<@�M�H?W=n@>�p%1<D�=p�C=UΒ=�h>�ѩ=�Q�p�@<�=��=�2�<Xig��&>ȏz�P�5���?=�4��ܹ=�J�=���.=�}=S�>=f��=���=}���dɠ�XN���pY=[W#=J˽|��<�E�=��ǽH��<� <��ɼ�	�(��a½t�(=��U����:pO�;�jn<��ڽ�(�@�� `,�d|h�5�v=8�H<�@����
�Mjѽ�Ǥ;X���8��=V��ɀ<xY ��:���,=�w�;ܓ��=�W��x< M����R��\����< �ѹ��������(�<�y��:��p�6<~k=Ȍ<�ݮ��T� ���
��"����.u�y�����  �=@Ʒ=�[���{��_�<��<%i��*�N=0�;t�<`�F=Q,��n���3O�\�=p�b<\NS����;���$�=y&��mi��l�D�@��;�����X����=6�k� c�E�l=��e=�3P<�=ǀ��۾ὼ��<YU���k�<pI��@����;�������<"��=��=�N�;��;y�l=X�M�@�;�h}���X=�$����|�P
Ż �^� u;$�=��= �u;�Q�l� �&�S=��W��=:Y�=�=�"���G<п7��B> �ܼ����>>J�5=�C����#�=V?:��i�x��<b��J����;�C��o����X<�{��6����z;m����������뇼�=<��}=Ъ���k�=g�@��==ˇ=fi �L���/�;Ȍa��� > ���j�=X�м��h<�V:����=�n|<,��=���=�.����ü'����;@���V5G���I;^��=�$>|�n=�i�<�Ln>d�+=�Z�Z�0= �}��>�Z����\=X�(� =κi=��=��?;�O�<b��!�R='�;v�=��<Ŋ�l]x�H<:R��P���HW�<>� =&�8=�^R=����a��$�=o����弮�=��M;-3�=��'�,��<d�@�:9=����h]<n��=�+�==�=]��=��h=|������< =Z�=H�V=�ܣ�k�=�I���7ӽL��<�qp����=�=�t�<
^="��=�3a=D��<)��=�\��Mh�ઝ���=,�=:F�;�<�JR=�����P���T������D�� ʂ����ܵ�<�y��6���2<؞=�sj�P���Y<���;xH���=t��<�B���<!��� ����|8�GD=h ��@t=��8�2�[��^=(�<�'�P� =<�$���_�ļ�<й+�xqƼd@=j<P{��:��,�<��ܼ�^�c�=�e�=pq<0�2��V�� ����N�4�������]J� Bͻ�ܘ=���=����齘��<�m�<������=��< :�<Д�<�.��R?1��炼�r�=��[� "o�̦��P����X=Kö�)���n���<��B���\��<�=��t�Ѐ�;�� =�!\=��<ڞI=}ڈ��Է�p ��Oc���}=(���=5=��9����; N�7�>灋=�֋<`�=�ݕ=x�<|�0���!�?w=����\������<x2<<ش�=�Ы=�2�<>���mڼϒ�=R�0����=��=u�=�̼x�<HM����>:�D����]>��<.�t�����U+����(����P=�TY�𵒼�o(���������+�<p���V�6�s��"�k�X�漭���@�j< I����<�=j��;=c &��J�=��==�~�1ۋ��d� bü�5�=țv��= A*��v��[�ʠ�=��<ң�=nw�=�>�p�P]��Z�;��!<��e�л�{��=i
�=XZ�=��=�iB>�l�ppG�`g<H$h����=�޼h�t=ȫt�0��� �;(s��D����g��.�_=����3�<Zm@=��������p���X��ڋ�Иr<�<��Q<��b= k��$��p��;�M�� �̹�)������s0=[D��
޻N�(� Љ<�A7;�,�<�cP=m"P=�M�=��y=�}=��<�Ǵ���G=؞�<�7}=`�ý�y?<�Gн@$罠��;�����c=Z=b=�x='SR=^��=y�9=dW޼�V%=������C�����f�;�i��P��լ<�k�;�a�"�o���V�`�һL�<@�<�a�� #M;`da��޼�4
<��/=�
�� .����^<�c�<`V��Hm=�*<]���T��<�� ��)=��+@���v<���;*�=ȕ0��*i��I=��;�[�;�[ =4ټ���G=l���@5$��:j={�=`�<<@���!= O�$���Eht=�X=T�;@�/���̺ \�8<���<H�����t��ʻ�,=���=��L�I[���pԻ��!���M�DW~=9� =xؤ<�Ɓ��œ�>�5��+<gs�=�[�!K=���,�< =��c����� �<��=(�#�����s�=V�K�Y�; �<�j=\�<��z=}F���jc�ꃼ� �)=p,��|s_=�`����:�@;\l>n��=P��<;�Z=�Ʉ=�5�<�l�pK�����=�W��Cɼ0Vi�`�;��<{D>Z��=��<�ͼ�~^�U�=�V� �c=4=L�<��5���O< ��;P��=��x�Tw���#=Ii=C��l�
��qԼ�� ��;�l��N�=�U<�m<�̟����;�G����;�ɳ� ��<��K=|�x=����)��/���e=�W+�!٫=�s���鄼�=�
���(���ļ��E=��F=�v"�T�<#.\=^GB=���;�p��י=c��\g= "t�@JQ��񗼫������4��<vC��T�ټ\�=�����>���8� �C<(�2<��;�����G=�
>�4���W=��*�HM༜��=��<,1a=�'<���;r�< �(�޼6����p=��<[m=8�{��)1��w�<\{��@�;��~���.��]�<Z[#�d�����Oѫ�P�g�P�F<|�ܼ�3��x<@�����:п��Z3`�Ђo�Z�=�k=JL�ޅ����=���/Ż�= �%< nU;d�=��׼��u���=��<(�)��X�=PW���r>=�Y?=�>�<x���»�C����=JWp��D=8̻�0��P?/��8���G��`i�$��<�������ji<=�RＰ���:$���!(���U��%*�H< @���ͦ��߻�FȽ�qm=ޗ���,=D�X9<N��h�=D����;��
=�Q"�h�|�f�_=�D�)�<�hK<���о�;���;8� =�\��^�v��Ë<��u�|U���hN=H�;<�G; ���r������U˽t�Y���6�9ɮ�&�D�h�=�i=�;!��f�� g<4����j� �'<Ԕ�<��<X(�<[潽�C���O��e=,��<PM���t����a<0u�<n���������< `���j<$p��na�=��c��@�:��=����� ����=y���S�׽��c�����s�<�W��tz�����;з��m�;�=ϼ��J;Ts��W�<Z�=��w��@��Nx=�*����l��&����z��=&�=�fD=����A
���C��+�,=�R���4=0��;�C�p��;+�!=L�ּa >@������ħB=p=ψ��|0ͼ ,X��S������09����D=(�D<��<Юλ��[��xS����� o�x��<��M=.iN��	��Zڼ�ϰ=�_�ƻ�=�m1��.�N=�@�3����
����<�0�=��j�<:=PJ�<�va=@w��cP���"=�޼�=��սLp���A��3������ ��:�Jн��=��=�q���� �;��w=��-=�w�:���:�(�=[7�=�*�� �-9Ȼ�<�����>�������<p�;�a8��h9=,A�� �
;=t���K�=�f�<��:@�<0FӼA�D=2C���刻�F��%��F�<:�2��@��@Ђ�n���ו���"��g�;m?Խ�!�<
��0�{<@��<�-W�X̡<͉y=��=J����y���C=P��Yü�&�;G��=p����=������V�=~@I=�yؼ$��<h�u<�V5=�Ó=P�<h�:���Ƽ,�!� ��=��U�#j<���L�ͼ�A��x�A<ܻ�������;�ӧ��[��'�F=�F���滨
!�R�wwɽ�Vo�T8���'�4�Ƚ ��`��;�L���u9=��z�N�/=`����<�(�`�';�H𼀥I<\��<(턼�����4Q=�-��h=@G<�6ɼ���;H�����<j���՞���<����x伤-=$Ò<��	:��+��L��(�Z�)ڽ��p���P��x���n�B�=9 I=zCK�=Z��˖<8����.�xzi< RN9���<0�<]�ɽz&�R�֔�=(��<�����;���;�r�<�Z�3P�� �Һ03�����;�G�6&�=;��Dj��V��=���< '����=9n����޽������"�<�O��0���;X*���<���=�e��2<h�/�x��<�$�<�̃�����＂=�Ū�H|��8T?�ܰ-���D=cz=�}= �ż����i���2=K�]��=���<0���=�;5dR=�׼h�>@/7���S�?o�=p��<ݯ���}��@=uz������v <p)�xS���<�Z �&�R�������|�@��8
m�hC'<;]ѽpLʽ���n0�=@�ּ$�>7��� �C)�=x����N=̑���T�=4�p��3c=$��X��=�u����w;@Z�T2�<��=o��B�ۗ��f%x����иټ�[�����=Zm!>��!� ���f>���=�=�7<���<�X�=1z=��-�e��j�<�ى<0+<>�����`�Z,4�j��:�=���<Ʋ<������m=�N0=�l� nP<�f��ڸ]=�R���8�+�N� ��Y�<������ڼd�g=��5��^�.��BM=ȠK�L4�<�p��4�<2��=�>`�~#�=8�=m��=Բ���X����<�t<,�߼d�]��� >:����V$=\�<J��ؤ�=���=�&��Tܥ<�~=6
�=���=]p=ҊQ�G,�������=Ld���ʋ� �9 �~<�p	�$��<�}):L���B�ꉩ��EԽL;=f�@� �:O�|S��-�ڽ4u��l
ԼH<��9Ľ8G<<��$<���=�d���$=�DS�R=�\)����T|̼���;@F�<@�;�/&���=�K+�X��<x<�x� l��p<�l5�<�,�wx����<,�%��)�\T�<�Ҷ<��2��#���� ��a�ս|!��<q�a~��F�j����=tj=�0i���� ��<�椼[��~�< �0���=��
=�{ӽFn��A�,{�=���<��(N�< �<;PU�<Rh�)V���1X����� ks:�m�����=X۰��G�dW|=�R=�DI;O2�=�+���E�p����R��y�<��Ƽ�� ���_%��=t�=�Q�;(�\<����T=�y:;0�������D��=4�Լ�㊽`/׻��*�@"=��=Ȝ=���uᑽ�?��Ga=z6H�WY�=�M= ��;��dh=@����>8\3�Pj����/>�c%=q⿽P@��Jh�=�?�gz��)�<�w̽�����W< <�8R��Ἶ=�u����{�:� j��sa��e������rٻ���=�n�J+�=�6��u��`>�=Ş��1����=Dw��>�!��=� >��#�=����&@=������=7�=���x�O�Aw���)K��a���H��3�^�>�3>(D��P<��p�b>��=@x�;~�=��=_y*>`�:;�b�;�:��=؍R=�&> ��7��M��㘽� a��{<���D��<�T�pڦ;�!Q=t��� �;��<��9=T�< ѷ:���,T�O�=�������`�=�Ű��
�9�A��f@=��f��a�<�f���!�<��=d_�<��=��=y��=�퉽��e�P��;Ԇf=�`�Du���.>>V���h+$�@X�<�]����=7��=�ނ�N�=��I=�B�=��=�@�=�Ȩ�:�>�P�i�
�d=��<�� � �;�hM=�� ��.�(�T���
�x�j�h��MQ��*�
=l���;� 1�pn�;~�a���i� ꑼXy<�o�0��<��<���*P<�����=;��]���=�CѼ���<@���*ռ�3=(�<X����<�^%���|���<�p��&r�V�=@i�<T�^�U�@�<��:b�f���<��<@S���h>�(]��$:9ؐ�\�|��*����8㮼�ǳ=2�=	<���ٽ��=��<�l��'=x <�g<(�<!�ýP���r�"��߱=�ș� s��0��;����H�<��۽�{˽�7v�@�� �L���=����=.�@�w;r?=��/=���;��v=E���=,ν�񸼳����<؆����\<Tx�Юü��&<��>���<P�<@��;Z�U=�)<��ru��#u�=|�x6�H#�� B�L4=�d�=3��=�K<��,���4���=�3+��ĝ=���<H�<�����<H)]����=,x��0B����;>7�F=����)��m�=L�.�p*!�v=�L�EB�x�><��μ$�༈�<�$
���T��0k���﬽@ȼh� ��c�е��#��=����X^�=?�D��h�:jU�= ���Ჽ,%I=�d��$��=E��<�=����#=᪽���=ҩ�4��=K}�=��н̟�:�h�d߼|��@�� ����>��%>�Y\< 2�� ހ>���=@���:=|A�<��+>�$�	�U=P�"���<�z= ��=�|�<��(<�g�T��<@�b;��3<x��<�����42=�y����$��=T��<���<<��<�*��'�#�^=�lp;�H%��^�=�;`͖<;�;��=�]G���;~gn��}=N��=��1=d��=�)�=D��=@� �ع< �����=xru<L׽��>\�`҈��K�<�M8��v�=�w�=���<�_=�Y=�[|=j��=��=��ؽ�A���X���3=KF=g/��7T<B33=ƽry/�(2
��� � ����l<Y��� 9�<�7��p�;��F<l= ��; ꃻ�mٻ�<Ь㼄J=
�2=e��� �q;��8��u估W��w�<0׏��l_=<{�z?d�O�G=�0< ����;��H�:���V<�z��x-Y��؏=��<�h��@��c�< _�<�g��V�<�m�<� ��G�<\߼@(6;4�ڼ��D��Mk�ږT�N<�i<=���=
U���� �=8s�<�V��W=��<�6� �/;����@̈́���Ƽ���=��{���:<�ӹ���� �=�+k���ýЧv<��<�W�P�m�5+W=��=�|8�<X�<t6'=��<��}=I ���-���-������?�< >¼�Y=�^R�@挻�iܻ)�>(B;=�ˁ<t&=�j=���<�����D���=Vw.�<Ê��Q�`��<��A=��=���=��<�A���G��̃�=��RqY=xt<<�dF<V�a��\� K����=�P����~�%>�=	��P6p��=���z��Ԧ,=��Ľn%� *0;^p�����i<<����ڇ�x�{����؞��������p^'�z�j= �2��1P=�-��a=��=�E��������r�X7}����=�ⱻ��=�چ� �f:�匽��d= D�����=���=*;��0���|m����4���k��웽\<��,��=ѥ�=d�V=�R�:��`>���<��G���<P@�;� >v�H�1Q|=���PS<��<t��<H4�<��';����,��< ��"�<LJ�<?�ݽ<�|��j�<&����¹���<ࠅ<@Km;iE-=���١�>�-=�,ڻ�h�ܵ�<п�;�K�<�7��E<�O$� _�T���(=n�=$+=�ݴ=���=b^W= F9P��;��Y<a�=~0=�ҽtq=^g����ƽ@q�;ad�� `=��3=�2=
�=�!�=�H=b�;=Ka6=ِ�ȡ���k� ��<�E�<w
��β<@��:h�}��k��h�e�tm�p��<)�'=ҘP���;@�U;�1�;x��<_�*=n�J=�~!<�i��N=�~���<�=��H�h�<B��(@����?�$6�<�R�=�r��j`�,�#= xQ;�n<�ϕ���ܼ�bW�(B�<`�?� G���Q�=(��<�*Y<p,�<���<�� =<������<`��;�+��С1<�$���<�?K<�ۼ0 /<�t̼�3< /<a�=�/���A�@�J;����?&��tn=MB=�T+�2����C�L}��@��:��B=nS����
=�)޼�[<��=�~�������=�=x (���!�(�<����i�<h���f8=0?<��m=ۗ��\�뼰��~A|�ز<�u��(q=��"��u��������=P[=�f�<��t=�4C=���<`AT<�t����X=z!.� =d<pX�d��<�)=�4�=�^�=�{=`�9� �
;��= �<��|�<����P�;+���0v����<�Œ=�}t��S�;*
dtype0*'
_output_shapes
:�
�
siamese_2/scala1/Conv2DConv2DPlaceholder_1 siamese/scala1/conv/weights/read*&
_output_shapes
:{{`*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_2/scala1/AddAddsiamese_2/scala1/Conv2Dsiamese/scala1/conv/biases/read*
T0*&
_output_shapes
:{{`
�
/siamese_2/scala1/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala1/moments/MeanMeansiamese_2/scala1/Add/siamese_2/scala1/moments/Mean/reduction_indices*
	keep_dims(*

Tidx0*
T0*&
_output_shapes
:`
�
%siamese_2/scala1/moments/StopGradientStopGradientsiamese_2/scala1/moments/Mean*
T0*&
_output_shapes
:`
y
4siamese_2/scala1/moments/sufficient_statistics/ConstConst*
dtype0*
_output_shapes
: *
valueB
 * K1G
�
2siamese_2/scala1/moments/sufficient_statistics/SubSubsiamese_2/scala1/Add%siamese_2/scala1/moments/StopGradient*&
_output_shapes
:{{`*
T0
�
@siamese_2/scala1/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese_2/scala1/Add%siamese_2/scala1/moments/StopGradient*
T0*&
_output_shapes
:{{`
�
Hsiamese_2/scala1/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese_2/scala1/moments/sufficient_statistics/mean_ssSum2siamese_2/scala1/moments/sufficient_statistics/SubHsiamese_2/scala1/moments/sufficient_statistics/mean_ss/reduction_indices*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:`
�
Gsiamese_2/scala1/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese_2/scala1/moments/sufficient_statistics/var_ssSum@siamese_2/scala1/moments/sufficient_statistics/SquaredDifferenceGsiamese_2/scala1/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes
:`*
	keep_dims( *

Tidx0
h
siamese_2/scala1/moments/ShapeConst*
valueB:`*
dtype0*
_output_shapes
:
�
 siamese_2/scala1/moments/ReshapeReshape%siamese_2/scala1/moments/StopGradientsiamese_2/scala1/moments/Shape*
T0*
Tshape0*
_output_shapes
:`
�
*siamese_2/scala1/moments/normalize/divisor
Reciprocal4siamese_2/scala1/moments/sufficient_statistics/Const7^siamese_2/scala1/moments/sufficient_statistics/mean_ss6^siamese_2/scala1/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
/siamese_2/scala1/moments/normalize/shifted_meanMul6siamese_2/scala1/moments/sufficient_statistics/mean_ss*siamese_2/scala1/moments/normalize/divisor*
_output_shapes
:`*
T0
�
'siamese_2/scala1/moments/normalize/meanAdd/siamese_2/scala1/moments/normalize/shifted_mean siamese_2/scala1/moments/Reshape*
T0*
_output_shapes
:`
�
&siamese_2/scala1/moments/normalize/MulMul5siamese_2/scala1/moments/sufficient_statistics/var_ss*siamese_2/scala1/moments/normalize/divisor*
_output_shapes
:`*
T0
�
)siamese_2/scala1/moments/normalize/SquareSquare/siamese_2/scala1/moments/normalize/shifted_mean*
_output_shapes
:`*
T0
�
+siamese_2/scala1/moments/normalize/varianceSub&siamese_2/scala1/moments/normalize/Mul)siamese_2/scala1/moments/normalize/Square*
_output_shapes
:`*
T0
�
&siamese_2/scala1/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/ConstConst*
_output_shapes
:`*
valueB`*    *0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/subSub8siamese/scala1/siamese/scala1/bn/moving_mean/biased/read'siamese_2/scala1/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mulMulBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub&siamese_2/scala1/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
ksiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_mean	AssignSub3siamese/scala1/siamese/scala1/bn/moving_mean/biasedBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
use_locking( *
T0
�
Nsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
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
Csiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readIdentity3siamese/scala1/siamese/scala1/bn/moving_mean/biasedl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/xConstl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1/x&siamese_2/scala1/AssignMovingAvg/decay*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: *
T0
�
Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1Identity7siamese/scala1/siamese/scala1/bn/moving_mean/local_stepl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/PowPowDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_1Esiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xConstl^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/siamese/scala1/siamese/scala1/bn/moving_meanI^siamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2SubFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2/xBsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/Pow*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
Fsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truedivRealDivCsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/readDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_2*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`*
T0
�
Dsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3Sub"siamese/scala1/bn/moving_mean/readFsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
 siamese_2/scala1/AssignMovingAvg	AssignSubsiamese/scala1/bn/moving_meanDsiamese_2/scala1/AssignMovingAvg/siamese/scala1/bn/moving_mean/sub_3*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean*
_output_shapes
:`
�
(siamese_2/scala1/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/ConstConst*
valueB`*    *4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
:`
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/subSub<siamese/scala1/siamese/scala1/bn/moving_variance/biased/read+siamese_2/scala1/moments/normalize/variance*
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
usiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_variance	AssignSub7siamese/scala1/siamese/scala1/bn/moving_variance/biasedHsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Tsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepTsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Isiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readIdentity7siamese/scala1/siamese/scala1/bn/moving_variance/biasedv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/xConstv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
dtype0
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1SubLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1/x(siamese_2/scala1/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1Identity;siamese/scala1/siamese/scala1/bn/moving_variance/local_stepv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/PowPowJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_1Ksiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/read_1*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
�
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2/xConstv^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/siamese/scala1/siamese/scala1/bn/moving_varianceO^siamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
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
Lsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truedivRealDivIsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/readJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
Jsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3Sub&siamese/scala1/bn/moving_variance/readLsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance*
_output_shapes
:`
�
"siamese_2/scala1/AssignMovingAvg_1	AssignSub!siamese/scala1/bn/moving_varianceJsiamese_2/scala1/AssignMovingAvg_1/siamese/scala1/bn/moving_variance/sub_3*
_output_shapes
:`*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala1/bn/moving_variance
c
siamese_2/scala1/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_2/scala1/cond/switch_tIdentitysiamese_2/scala1/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_2/scala1/cond/switch_fIdentitysiamese_2/scala1/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_2/scala1/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_2/scala1/cond/Switch_1Switch'siamese_2/scala1/moments/normalize/meansiamese_2/scala1/cond/pred_id*
T0*:
_class0
.,loc:@siamese_2/scala1/moments/normalize/mean* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_2Switch+siamese_2/scala1/moments/normalize/variancesiamese_2/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*>
_class4
20loc:@siamese_2/scala1/moments/normalize/variance
�
%siamese_2/scala1/cond/Switch_3/SwitchSwitch"siamese/scala1/bn/moving_mean/readsiamese_2/scala1/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean* 
_output_shapes
:`:`
�
siamese_2/scala1/cond/Switch_3Switch%siamese_2/scala1/cond/Switch_3/Switchsiamese_2/scala1/cond/pred_id* 
_output_shapes
:`:`*
T0*0
_class&
$"loc:@siamese/scala1/bn/moving_mean
�
%siamese_2/scala1/cond/Switch_4/SwitchSwitch&siamese/scala1/bn/moving_variance/readsiamese_2/scala1/cond/pred_id*4
_class*
(&loc:@siamese/scala1/bn/moving_variance* 
_output_shapes
:`:`*
T0
�
siamese_2/scala1/cond/Switch_4Switch%siamese_2/scala1/cond/Switch_4/Switchsiamese_2/scala1/cond/pred_id*
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
siamese_2/scala1/batchnorm/addAddsiamese_2/scala1/cond/Merge_1 siamese_2/scala1/batchnorm/add/y*
T0*
_output_shapes
:`
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
 siamese_2/scala1/batchnorm/mul_1Mulsiamese_2/scala1/Addsiamese_2/scala1/batchnorm/mul*&
_output_shapes
:{{`*
T0
�
 siamese_2/scala1/batchnorm/mul_2Mulsiamese_2/scala1/cond/Mergesiamese_2/scala1/batchnorm/mul*
T0*
_output_shapes
:`
�
siamese_2/scala1/batchnorm/subSubsiamese/scala1/bn/beta/read siamese_2/scala1/batchnorm/mul_2*
_output_shapes
:`*
T0
�
 siamese_2/scala1/batchnorm/add_1Add siamese_2/scala1/batchnorm/mul_1siamese_2/scala1/batchnorm/sub*
T0*&
_output_shapes
:{{`
p
siamese_2/scala1/ReluRelu siamese_2/scala1/batchnorm/add_1*
T0*&
_output_shapes
:{{`
�
siamese_2/scala1/poll/MaxPoolMaxPoolsiamese_2/scala1/Relu*&
_output_shapes
:==`*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
b
 siamese_2/scala2/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/splitSplit siamese_2/scala2/split/split_dimsiamese_2/scala1/poll/MaxPool*8
_output_shapes&
$:==0:==0*
	num_split*
T0
d
"siamese_2/scala2/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/split_1Split"siamese_2/scala2/split_1/split_dim siamese/scala2/conv/weights/read*:
_output_shapes(
&:0�:0�*
	num_split*
T0
�
siamese_2/scala2/Conv2DConv2Dsiamese_2/scala2/splitsiamese_2/scala2/split_1*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:99�*
T0*
data_formatNHWC*
strides

�
siamese_2/scala2/Conv2D_1Conv2Dsiamese_2/scala2/split:1siamese_2/scala2/split_1:1*'
_output_shapes
:99�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_2/scala2/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/concatConcatV2siamese_2/scala2/Conv2Dsiamese_2/scala2/Conv2D_1siamese_2/scala2/concat/axis*
T0*
N*'
_output_shapes
:99�*

Tidx0
�
siamese_2/scala2/AddAddsiamese_2/scala2/concatsiamese/scala2/conv/biases/read*
T0*'
_output_shapes
:99�
�
/siamese_2/scala2/moments/Mean/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
siamese_2/scala2/moments/MeanMeansiamese_2/scala2/Add/siamese_2/scala2/moments/Mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_2/scala2/moments/StopGradientStopGradientsiamese_2/scala2/moments/Mean*
T0*'
_output_shapes
:�
y
4siamese_2/scala2/moments/sufficient_statistics/ConstConst*
valueB
 * LF*
dtype0*
_output_shapes
: 
�
2siamese_2/scala2/moments/sufficient_statistics/SubSubsiamese_2/scala2/Add%siamese_2/scala2/moments/StopGradient*
T0*'
_output_shapes
:99�
�
@siamese_2/scala2/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese_2/scala2/Add%siamese_2/scala2/moments/StopGradient*
T0*'
_output_shapes
:99�
�
Hsiamese_2/scala2/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese_2/scala2/moments/sufficient_statistics/mean_ssSum2siamese_2/scala2/moments/sufficient_statistics/SubHsiamese_2/scala2/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
Gsiamese_2/scala2/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese_2/scala2/moments/sufficient_statistics/var_ssSum@siamese_2/scala2/moments/sufficient_statistics/SquaredDifferenceGsiamese_2/scala2/moments/sufficient_statistics/var_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
i
siamese_2/scala2/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
 siamese_2/scala2/moments/ReshapeReshape%siamese_2/scala2/moments/StopGradientsiamese_2/scala2/moments/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
*siamese_2/scala2/moments/normalize/divisor
Reciprocal4siamese_2/scala2/moments/sufficient_statistics/Const7^siamese_2/scala2/moments/sufficient_statistics/mean_ss6^siamese_2/scala2/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
/siamese_2/scala2/moments/normalize/shifted_meanMul6siamese_2/scala2/moments/sufficient_statistics/mean_ss*siamese_2/scala2/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
'siamese_2/scala2/moments/normalize/meanAdd/siamese_2/scala2/moments/normalize/shifted_mean siamese_2/scala2/moments/Reshape*
T0*
_output_shapes	
:�
�
&siamese_2/scala2/moments/normalize/MulMul5siamese_2/scala2/moments/sufficient_statistics/var_ss*siamese_2/scala2/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
)siamese_2/scala2/moments/normalize/SquareSquare/siamese_2/scala2/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
+siamese_2/scala2/moments/normalize/varianceSub&siamese_2/scala2/moments/normalize/Mul)siamese_2/scala2/moments/normalize/Square*
T0*
_output_shapes	
:�
�
&siamese_2/scala2/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes	
:�
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/subSub8siamese/scala2/siamese/scala2/bn/moving_mean/biased/read'siamese_2/scala2/moments/normalize/mean*
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
ksiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_mean	AssignSub3siamese/scala2/siamese/scala2/bn/moving_mean/biasedBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/mul*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Nsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/valueConst*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Hsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepNsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd/value*
_output_shapes
: *
use_locking( *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Csiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readIdentity3siamese/scala2/siamese/scala2/bn/moving_mean/biasedl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1/xConstl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
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
Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1Identity7siamese/scala2/siamese/scala2/bn/moving_mean/local_stepl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Bsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/PowPowDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_1Esiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/read_1*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xConstl^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/siamese/scala2/siamese/scala2/bn/moving_meanI^siamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/AssignAdd*
dtype0*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala2/bn/moving_mean
�
Dsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2SubFsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2/xBsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/truedivRealDivCsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/readDsiamese_2/scala2/AssignMovingAvg/siamese/scala2/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*
_output_shapes	
:�
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
(siamese_2/scala2/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/subSub<siamese/scala2/siamese/scala2/bn/moving_variance/biased/read+siamese_2/scala2/moments/normalize/variance*
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
usiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_variance	AssignSub7siamese/scala2/siamese/scala2/bn/moving_variance/biasedHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
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
Isiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readIdentity7siamese/scala2/siamese/scala2/bn/moving_variance/biasedv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/xConstv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1SubLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_1/x(siamese_2/scala2/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Ksiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/read_1Identity;siamese/scala2/siamese/scala2/bn/moving_variance/local_stepv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
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
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xConstv^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/siamese/scala2/siamese/scala2/bn/moving_varianceO^siamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2SubLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2/xHsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/Pow*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truedivRealDivIsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/readJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3Sub&siamese/scala2/bn/moving_variance/readLsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/truediv*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�
�
"siamese_2/scala2/AssignMovingAvg_1	AssignSub!siamese/scala2/bn/moving_varianceJsiamese_2/scala2/AssignMovingAvg_1/siamese/scala2/bn/moving_variance/sub_3*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*
_output_shapes	
:�*
use_locking( 
c
siamese_2/scala2/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_2/scala2/cond/switch_tIdentitysiamese_2/scala2/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_2/scala2/cond/switch_fIdentitysiamese_2/scala2/cond/Switch*
_output_shapes
: *
T0

W
siamese_2/scala2/cond/pred_idIdentityis_training*
T0
*
_output_shapes
: 
�
siamese_2/scala2/cond/Switch_1Switch'siamese_2/scala2/moments/normalize/meansiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*:
_class0
.,loc:@siamese_2/scala2/moments/normalize/mean
�
siamese_2/scala2/cond/Switch_2Switch+siamese_2/scala2/moments/normalize/variancesiamese_2/scala2/cond/pred_id*
T0*>
_class4
20loc:@siamese_2/scala2/moments/normalize/variance*"
_output_shapes
:�:�
�
%siamese_2/scala2/cond/Switch_3/SwitchSwitch"siamese/scala2/bn/moving_mean/readsiamese_2/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/Switch_3Switch%siamese_2/scala2/cond/Switch_3/Switchsiamese_2/scala2/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala2/bn/moving_mean*"
_output_shapes
:�:�
�
%siamese_2/scala2/cond/Switch_4/SwitchSwitch&siamese/scala2/bn/moving_variance/readsiamese_2/scala2/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_2/scala2/cond/Switch_4Switch%siamese_2/scala2/cond/Switch_4/Switchsiamese_2/scala2/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala2/bn/moving_variance
�
siamese_2/scala2/cond/MergeMergesiamese_2/scala2/cond/Switch_3 siamese_2/scala2/cond/Switch_1:1*
T0*
N*
_output_shapes
	:�: 
�
siamese_2/scala2/cond/Merge_1Mergesiamese_2/scala2/cond/Switch_4 siamese_2/scala2/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_2/scala2/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala2/batchnorm/addAddsiamese_2/scala2/cond/Merge_1 siamese_2/scala2/batchnorm/add/y*
T0*
_output_shapes	
:�
o
 siamese_2/scala2/batchnorm/RsqrtRsqrtsiamese_2/scala2/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_2/scala2/batchnorm/mulMul siamese_2/scala2/batchnorm/Rsqrtsiamese/scala2/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_2/scala2/batchnorm/mul_1Mulsiamese_2/scala2/Addsiamese_2/scala2/batchnorm/mul*
T0*'
_output_shapes
:99�
�
 siamese_2/scala2/batchnorm/mul_2Mulsiamese_2/scala2/cond/Mergesiamese_2/scala2/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_2/scala2/batchnorm/subSubsiamese/scala2/bn/beta/read siamese_2/scala2/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_2/scala2/batchnorm/add_1Add siamese_2/scala2/batchnorm/mul_1siamese_2/scala2/batchnorm/sub*
T0*'
_output_shapes
:99�
q
siamese_2/scala2/ReluRelu siamese_2/scala2/batchnorm/add_1*'
_output_shapes
:99�*
T0
�
siamese_2/scala2/poll/MaxPoolMaxPoolsiamese_2/scala2/Relu*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
ksize
*
paddingVALID
�
siamese_2/scala3/Conv2DConv2Dsiamese_2/scala2/poll/MaxPool siamese/scala3/conv/weights/read*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_2/scala3/AddAddsiamese_2/scala3/Conv2Dsiamese/scala3/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_2/scala3/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala3/moments/MeanMeansiamese_2/scala3/Add/siamese_2/scala3/moments/Mean/reduction_indices*
T0*'
_output_shapes
:�*
	keep_dims(*

Tidx0
�
%siamese_2/scala3/moments/StopGradientStopGradientsiamese_2/scala3/moments/Mean*
T0*'
_output_shapes
:�
y
4siamese_2/scala3/moments/sufficient_statistics/ConstConst*
valueB
 * ��D*
dtype0*
_output_shapes
: 
�
2siamese_2/scala3/moments/sufficient_statistics/SubSubsiamese_2/scala3/Add%siamese_2/scala3/moments/StopGradient*'
_output_shapes
:�*
T0
�
@siamese_2/scala3/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese_2/scala3/Add%siamese_2/scala3/moments/StopGradient*'
_output_shapes
:�*
T0
�
Hsiamese_2/scala3/moments/sufficient_statistics/mean_ss/reduction_indicesConst*
dtype0*
_output_shapes
:*!
valueB"          
�
6siamese_2/scala3/moments/sufficient_statistics/mean_ssSum2siamese_2/scala3/moments/sufficient_statistics/SubHsiamese_2/scala3/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
Gsiamese_2/scala3/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese_2/scala3/moments/sufficient_statistics/var_ssSum@siamese_2/scala3/moments/sufficient_statistics/SquaredDifferenceGsiamese_2/scala3/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
i
siamese_2/scala3/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
 siamese_2/scala3/moments/ReshapeReshape%siamese_2/scala3/moments/StopGradientsiamese_2/scala3/moments/Shape*
_output_shapes	
:�*
T0*
Tshape0
�
*siamese_2/scala3/moments/normalize/divisor
Reciprocal4siamese_2/scala3/moments/sufficient_statistics/Const7^siamese_2/scala3/moments/sufficient_statistics/mean_ss6^siamese_2/scala3/moments/sufficient_statistics/var_ss*
T0*
_output_shapes
: 
�
/siamese_2/scala3/moments/normalize/shifted_meanMul6siamese_2/scala3/moments/sufficient_statistics/mean_ss*siamese_2/scala3/moments/normalize/divisor*
_output_shapes	
:�*
T0
�
'siamese_2/scala3/moments/normalize/meanAdd/siamese_2/scala3/moments/normalize/shifted_mean siamese_2/scala3/moments/Reshape*
T0*
_output_shapes	
:�
�
&siamese_2/scala3/moments/normalize/MulMul5siamese_2/scala3/moments/sufficient_statistics/var_ss*siamese_2/scala3/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
)siamese_2/scala3/moments/normalize/SquareSquare/siamese_2/scala3/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
+siamese_2/scala3/moments/normalize/varianceSub&siamese_2/scala3/moments/normalize/Mul)siamese_2/scala3/moments/normalize/Square*
_output_shapes	
:�*
T0
�
&siamese_2/scala3/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0*
_output_shapes	
:�
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/subSub8siamese/scala3/siamese/scala3/bn/moving_mean/biased/read'siamese_2/scala3/moments/normalize/mean*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
T0
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mulMulBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub&siamese_2/scala3/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_mean	AssignSub3siamese/scala3/siamese/scala3/bn/moving_mean/biasedBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/mul*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( 
�
Nsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/valueConst*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0
�
Hsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepNsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readIdentity3siamese/scala3/siamese/scala3/bn/moving_mean/biasedl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/xConstl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1/x&siamese_2/scala3/AssignMovingAvg/decay*
_output_shapes
: *
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1Identity7siamese/scala3/siamese/scala3/bn/moving_mean/local_stepl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Bsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/PowPowDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_1Esiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/read_1*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: *
T0
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xConstl^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/siamese/scala3/siamese/scala3/bn/moving_meanI^siamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/AssignAdd*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
dtype0
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2SubFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2/xBsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes
: 
�
Fsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truedivRealDivCsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/readDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_2*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
Dsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3Sub"siamese/scala3/bn/moving_mean/readFsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/truediv*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�
�
 siamese_2/scala3/AssignMovingAvg	AssignSubsiamese/scala3/bn/moving_meanDsiamese_2/scala3/AssignMovingAvg/siamese/scala3/bn/moving_mean/sub_3*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
(siamese_2/scala3/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/ConstConst*
dtype0*
_output_shapes	
:�*
valueB�*    *4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/subSub<siamese/scala3/siamese/scala3/bn/moving_variance/biased/read+siamese_2/scala3/moments/normalize/variance*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mulMulHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub(siamese_2/scala3/AssignMovingAvg_1/decay*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
usiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_variance	AssignSub7siamese/scala3/siamese/scala3/bn/moving_variance/biasedHsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/mul*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Tsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepTsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd/value*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: *
use_locking( *
T0
�
Isiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/readIdentity7siamese/scala3/siamese/scala3/bn/moving_variance/biasedv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/xConstv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
_output_shapes
: *
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
dtype0
�
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1SubLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1/x(siamese_2/scala3/AssignMovingAvg_1/decay*
_output_shapes
: *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1Identity;siamese/scala3/siamese/scala3/bn/moving_variance/local_stepv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Hsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/PowPowJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_1Ksiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/read_1*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes
: 
�
Lsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_2/xConstv^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/siamese/scala3/siamese/scala3/bn/moving_varianceO^siamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
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
Jsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3Sub&siamese/scala3/bn/moving_variance/readLsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese_2/scala3/AssignMovingAvg_1	AssignSub!siamese/scala3/bn/moving_varianceJsiamese_2/scala3/AssignMovingAvg_1/siamese/scala3/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*
_output_shapes	
:�
c
siamese_2/scala3/cond/SwitchSwitchis_trainingis_training*
_output_shapes
: : *
T0

k
siamese_2/scala3/cond/switch_tIdentitysiamese_2/scala3/cond/Switch:1*
_output_shapes
: *
T0

i
siamese_2/scala3/cond/switch_fIdentitysiamese_2/scala3/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_2/scala3/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_2/scala3/cond/Switch_1Switch'siamese_2/scala3/moments/normalize/meansiamese_2/scala3/cond/pred_id*
T0*:
_class0
.,loc:@siamese_2/scala3/moments/normalize/mean*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/Switch_2Switch+siamese_2/scala3/moments/normalize/variancesiamese_2/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*>
_class4
20loc:@siamese_2/scala3/moments/normalize/variance
�
%siamese_2/scala3/cond/Switch_3/SwitchSwitch"siamese/scala3/bn/moving_mean/readsiamese_2/scala3/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/Switch_3Switch%siamese_2/scala3/cond/Switch_3/Switchsiamese_2/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*0
_class&
$"loc:@siamese/scala3/bn/moving_mean
�
%siamese_2/scala3/cond/Switch_4/SwitchSwitch&siamese/scala3/bn/moving_variance/readsiamese_2/scala3/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance
�
siamese_2/scala3/cond/Switch_4Switch%siamese_2/scala3/cond/Switch_4/Switchsiamese_2/scala3/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala3/bn/moving_variance*"
_output_shapes
:�:�
�
siamese_2/scala3/cond/MergeMergesiamese_2/scala3/cond/Switch_3 siamese_2/scala3/cond/Switch_1:1*
_output_shapes
	:�: *
T0*
N
�
siamese_2/scala3/cond/Merge_1Mergesiamese_2/scala3/cond/Switch_4 siamese_2/scala3/cond/Switch_2:1*
T0*
N*
_output_shapes
	:�: 
e
 siamese_2/scala3/batchnorm/add/yConst*
valueB
 *o�:*
dtype0*
_output_shapes
: 
�
siamese_2/scala3/batchnorm/addAddsiamese_2/scala3/cond/Merge_1 siamese_2/scala3/batchnorm/add/y*
_output_shapes	
:�*
T0
o
 siamese_2/scala3/batchnorm/RsqrtRsqrtsiamese_2/scala3/batchnorm/add*
T0*
_output_shapes	
:�
�
siamese_2/scala3/batchnorm/mulMul siamese_2/scala3/batchnorm/Rsqrtsiamese/scala3/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_2/scala3/batchnorm/mul_1Mulsiamese_2/scala3/Addsiamese_2/scala3/batchnorm/mul*'
_output_shapes
:�*
T0
�
 siamese_2/scala3/batchnorm/mul_2Mulsiamese_2/scala3/cond/Mergesiamese_2/scala3/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_2/scala3/batchnorm/subSubsiamese/scala3/bn/beta/read siamese_2/scala3/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_2/scala3/batchnorm/add_1Add siamese_2/scala3/batchnorm/mul_1siamese_2/scala3/batchnorm/sub*'
_output_shapes
:�*
T0
q
siamese_2/scala3/ReluRelu siamese_2/scala3/batchnorm/add_1*
T0*'
_output_shapes
:�
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
&:�:�*
	num_split
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
siamese_2/scala4/Conv2DConv2Dsiamese_2/scala4/splitsiamese_2/scala4/split_1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_2/scala4/Conv2D_1Conv2Dsiamese_2/scala4/split:1siamese_2/scala4/split_1:1*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*'
_output_shapes
:�
^
siamese_2/scala4/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala4/concatConcatV2siamese_2/scala4/Conv2Dsiamese_2/scala4/Conv2D_1siamese_2/scala4/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:�
�
siamese_2/scala4/AddAddsiamese_2/scala4/concatsiamese/scala4/conv/biases/read*
T0*'
_output_shapes
:�
�
/siamese_2/scala4/moments/Mean/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
siamese_2/scala4/moments/MeanMeansiamese_2/scala4/Add/siamese_2/scala4/moments/Mean/reduction_indices*'
_output_shapes
:�*
	keep_dims(*

Tidx0*
T0
�
%siamese_2/scala4/moments/StopGradientStopGradientsiamese_2/scala4/moments/Mean*'
_output_shapes
:�*
T0
y
4siamese_2/scala4/moments/sufficient_statistics/ConstConst*
valueB
 *  �D*
dtype0*
_output_shapes
: 
�
2siamese_2/scala4/moments/sufficient_statistics/SubSubsiamese_2/scala4/Add%siamese_2/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
�
@siamese_2/scala4/moments/sufficient_statistics/SquaredDifferenceSquaredDifferencesiamese_2/scala4/Add%siamese_2/scala4/moments/StopGradient*
T0*'
_output_shapes
:�
�
Hsiamese_2/scala4/moments/sufficient_statistics/mean_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
6siamese_2/scala4/moments/sufficient_statistics/mean_ssSum2siamese_2/scala4/moments/sufficient_statistics/SubHsiamese_2/scala4/moments/sufficient_statistics/mean_ss/reduction_indices*
_output_shapes	
:�*
	keep_dims( *

Tidx0*
T0
�
Gsiamese_2/scala4/moments/sufficient_statistics/var_ss/reduction_indicesConst*!
valueB"          *
dtype0*
_output_shapes
:
�
5siamese_2/scala4/moments/sufficient_statistics/var_ssSum@siamese_2/scala4/moments/sufficient_statistics/SquaredDifferenceGsiamese_2/scala4/moments/sufficient_statistics/var_ss/reduction_indices*
T0*
_output_shapes	
:�*
	keep_dims( *

Tidx0
i
siamese_2/scala4/moments/ShapeConst*
valueB:�*
dtype0*
_output_shapes
:
�
 siamese_2/scala4/moments/ReshapeReshape%siamese_2/scala4/moments/StopGradientsiamese_2/scala4/moments/Shape*
T0*
Tshape0*
_output_shapes	
:�
�
*siamese_2/scala4/moments/normalize/divisor
Reciprocal4siamese_2/scala4/moments/sufficient_statistics/Const7^siamese_2/scala4/moments/sufficient_statistics/mean_ss6^siamese_2/scala4/moments/sufficient_statistics/var_ss*
_output_shapes
: *
T0
�
/siamese_2/scala4/moments/normalize/shifted_meanMul6siamese_2/scala4/moments/sufficient_statistics/mean_ss*siamese_2/scala4/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
'siamese_2/scala4/moments/normalize/meanAdd/siamese_2/scala4/moments/normalize/shifted_mean siamese_2/scala4/moments/Reshape*
T0*
_output_shapes	
:�
�
&siamese_2/scala4/moments/normalize/MulMul5siamese_2/scala4/moments/sufficient_statistics/var_ss*siamese_2/scala4/moments/normalize/divisor*
T0*
_output_shapes	
:�
�
)siamese_2/scala4/moments/normalize/SquareSquare/siamese_2/scala4/moments/normalize/shifted_mean*
T0*
_output_shapes	
:�
�
+siamese_2/scala4/moments/normalize/varianceSub&siamese_2/scala4/moments/normalize/Mul)siamese_2/scala4/moments/normalize/Square*
_output_shapes	
:�*
T0
�
&siamese_2/scala4/AssignMovingAvg/decayConst*
valueB
 *RI�9*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes
: 
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/ConstConst*
valueB�*    *0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0*
_output_shapes	
:�
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/subSub8siamese/scala4/siamese/scala4/bn/moving_mean/biased/read'siamese_2/scala4/moments/normalize/mean*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Bsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mulMulBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub&siamese_2/scala4/AssignMovingAvg/decay*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
ksiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_mean	AssignSub3siamese/scala4/siamese/scala4/bn/moving_mean/biasedBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/mul*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�*
use_locking( *
T0
�
Nsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/valueConst*
dtype0*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
Hsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd	AssignAdd7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepNsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd/value*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
�
Csiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/readIdentity3siamese/scala4/siamese/scala4/bn/moving_mean/biasedl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes	
:�
�
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_1/xConstl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
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
Esiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/read_1Identity7siamese/scala4/siamese/scala4/bn/moving_mean/local_stepl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
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
Fsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xConstl^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/siamese/scala4/siamese/scala4/bn/moving_meanI^siamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/AssignAdd*
_output_shapes
: *
valueB
 *  �?*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
dtype0
�
Dsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2SubFsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_2/xBsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/Pow*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*
_output_shapes
: 
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
 siamese_2/scala4/AssignMovingAvg	AssignSubsiamese/scala4/bn/moving_meanDsiamese_2/scala4/AssignMovingAvg/siamese/scala4/bn/moving_mean/sub_3*
_output_shapes	
:�*
use_locking( *
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean
�
(siamese_2/scala4/AssignMovingAvg_1/decayConst*
valueB
 *RI�9*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/ConstConst*
valueB�*    *4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes	
:�
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/subSub<siamese/scala4/siamese/scala4/bn/moving_variance/biased/read+siamese_2/scala4/moments/normalize/variance*
_output_shapes	
:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
Hsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mulMulHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub(siamese_2/scala4/AssignMovingAvg_1/decay*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
usiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_variance	AssignSub7siamese/scala4/siamese/scala4/bn/moving_variance/biasedHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/mul*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
use_locking( *
T0
�
Tsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/valueConst*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Nsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd	AssignAdd;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepTsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd/value*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: 
�
Isiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readIdentity7siamese/scala4/siamese/scala4/bn/moving_variance/biasedv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_1/xConstv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
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
Ksiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/read_1Identity;siamese/scala4/siamese/scala4/bn/moving_variance/local_stepv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
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
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xConstv^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/siamese/scala4/siamese/scala4/bn/moving_varianceO^siamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/AssignAdd*
valueB
 *  �?*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
dtype0*
_output_shapes
: 
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2SubLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2/xHsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/Pow*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes
: *
T0
�
Lsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truedivRealDivIsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/readJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_2*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
�
Jsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3Sub&siamese/scala4/bn/moving_variance/readLsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/truediv*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�*
T0
�
"siamese_2/scala4/AssignMovingAvg_1	AssignSub!siamese/scala4/bn/moving_varianceJsiamese_2/scala4/AssignMovingAvg_1/siamese/scala4/bn/moving_variance/sub_3*
use_locking( *
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*
_output_shapes	
:�
c
siamese_2/scala4/cond/SwitchSwitchis_trainingis_training*
T0
*
_output_shapes
: : 
k
siamese_2/scala4/cond/switch_tIdentitysiamese_2/scala4/cond/Switch:1*
T0
*
_output_shapes
: 
i
siamese_2/scala4/cond/switch_fIdentitysiamese_2/scala4/cond/Switch*
T0
*
_output_shapes
: 
W
siamese_2/scala4/cond/pred_idIdentityis_training*
_output_shapes
: *
T0

�
siamese_2/scala4/cond/Switch_1Switch'siamese_2/scala4/moments/normalize/meansiamese_2/scala4/cond/pred_id*:
_class0
.,loc:@siamese_2/scala4/moments/normalize/mean*"
_output_shapes
:�:�*
T0
�
siamese_2/scala4/cond/Switch_2Switch+siamese_2/scala4/moments/normalize/variancesiamese_2/scala4/cond/pred_id*
T0*>
_class4
20loc:@siamese_2/scala4/moments/normalize/variance*"
_output_shapes
:�:�
�
%siamese_2/scala4/cond/Switch_3/SwitchSwitch"siamese/scala4/bn/moving_mean/readsiamese_2/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
siamese_2/scala4/cond/Switch_3Switch%siamese_2/scala4/cond/Switch_3/Switchsiamese_2/scala4/cond/pred_id*
T0*0
_class&
$"loc:@siamese/scala4/bn/moving_mean*"
_output_shapes
:�:�
�
%siamese_2/scala4/cond/Switch_4/SwitchSwitch&siamese/scala4/bn/moving_variance/readsiamese_2/scala4/cond/pred_id*"
_output_shapes
:�:�*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance
�
siamese_2/scala4/cond/Switch_4Switch%siamese_2/scala4/cond/Switch_4/Switchsiamese_2/scala4/cond/pred_id*
T0*4
_class*
(&loc:@siamese/scala4/bn/moving_variance*"
_output_shapes
:�:�
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
 siamese_2/scala4/batchnorm/RsqrtRsqrtsiamese_2/scala4/batchnorm/add*
_output_shapes	
:�*
T0
�
siamese_2/scala4/batchnorm/mulMul siamese_2/scala4/batchnorm/Rsqrtsiamese/scala4/bn/gamma/read*
_output_shapes	
:�*
T0
�
 siamese_2/scala4/batchnorm/mul_1Mulsiamese_2/scala4/Addsiamese_2/scala4/batchnorm/mul*
T0*'
_output_shapes
:�
�
 siamese_2/scala4/batchnorm/mul_2Mulsiamese_2/scala4/cond/Mergesiamese_2/scala4/batchnorm/mul*
T0*
_output_shapes	
:�
�
siamese_2/scala4/batchnorm/subSubsiamese/scala4/bn/beta/read siamese_2/scala4/batchnorm/mul_2*
_output_shapes	
:�*
T0
�
 siamese_2/scala4/batchnorm/add_1Add siamese_2/scala4/batchnorm/mul_1siamese_2/scala4/batchnorm/sub*
T0*'
_output_shapes
:�
q
siamese_2/scala4/ReluRelu siamese_2/scala4/batchnorm/add_1*
T0*'
_output_shapes
:�
b
 siamese_2/scala5/split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0
�
siamese_2/scala5/splitSplit siamese_2/scala5/split/split_dimsiamese_2/scala4/Relu*
T0*:
_output_shapes(
&:�:�*
	num_split
d
"siamese_2/scala5/split_1/split_dimConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/split_1Split"siamese_2/scala5/split_1/split_dim siamese/scala5/conv/weights/read*<
_output_shapes*
(:��:��*
	num_split*
T0
�
siamese_2/scala5/Conv2DConv2Dsiamese_2/scala5/splitsiamese_2/scala5/split_1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
�
siamese_2/scala5/Conv2D_1Conv2Dsiamese_2/scala5/split:1siamese_2/scala5/split_1:1*'
_output_shapes
:�*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
^
siamese_2/scala5/concat/axisConst*
value	B :*
dtype0*
_output_shapes
: 
�
siamese_2/scala5/concatConcatV2siamese_2/scala5/Conv2Dsiamese_2/scala5/Conv2D_1siamese_2/scala5/concat/axis*'
_output_shapes
:�*

Tidx0*
T0*
N
�
siamese_2/scala5/AddAddsiamese_2/scala5/concatsiamese/scala5/conv/biases/read*'
_output_shapes
:�*
T0
Y
score_1/split/split_dimConst*
value	B : *
dtype0*
_output_shapes
: 
�
score_1/splitSplitscore_1/split/split_dimsiamese_2/scala5/Add*M
_output_shapes;
9:�:�:�*
	num_split*
T0
�
score_1/Conv2DConv2Dscore_1/splitConst*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score_1/Conv2D_1Conv2Dscore_1/split:1Const*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:
�
score_1/Conv2D_2Conv2Dscore_1/split:2Const*
use_cudnn_on_gpu(*
paddingVALID*&
_output_shapes
:*
T0*
data_formatNHWC*
strides

U
score_1/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
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
:*
T0*
data_formatNHWC*
strides
*
use_cudnn_on_gpu(*
paddingVALID
i
adjust_1/AddAddadjust_1/Conv2Dadjust/biases/read*
T0*&
_output_shapes
:"��"N