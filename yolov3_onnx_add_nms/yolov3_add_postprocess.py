import numpy as np
import onnx
from onnx import defs, checker, helper, numpy_helper, mapping
from onnx import ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto, OperatorProto, OperatorSetIdProto
from onnx.helper import make_tensor, make_tensor_value_info, make_attribute, make_model, make_node

def define_a_new_graph(graph, unused_node=[]):

    ngraph = GraphProto()
    ngraph.name = graph.name

    ngraph.input.extend([i for i in graph.input if i.name not in unused_node])
    ngraph.initializer.extend([i for i in graph.initializer if i.name not in unused_node])
    ngraph.value_info.extend([i for i in graph.value_info if i.name not in unused_node])
    ngraph.node.extend([i for i in graph.node if i.name not in unused_node])

    output_info = [i for i in graph.output]
    ngraph.value_info.extend(output_info)
    print("graph input is:", graph.input[0])
    print(graph.output)
    origin_input_resolution = make_tensor_value_info("origin_input_resolution", TensorProto.FLOAT, ["-1", 1, 4]) # -1代表batch, 4代表[w, h, w, h]

    ########## Grid0: -1x255x19x19 ##########
    grid0_transpose_0_op_output0 = make_tensor_value_info("grid0_transpose_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 255])
    grid0_transpose_0_op = make_node(
        "Transpose", 
        perm=[0, 2, 3, 1], 
        inputs=["output0"],
        outputs=["grid0_transpose_0_op_output0"])

    grid0_reshape_0_op_shape = make_tensor_value_info("grid0_reshape_0_op_shape", TensorProto.INT64, [1])
    grid0_reshape_0_op_shape_init = numpy_helper.from_array(np.array([-1, 19, 19, 3, 85], dtype=np.int64))
    grid0_reshape_0_op_shape_init.name = "grid0_reshape_0_op_shape"

    grid0_reshape_0_op_output0 = make_tensor_value_info("grid0_reshape_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 85])
    grid0_reshape_0_op = make_node(
        "Reshape",
        inputs=["grid0_transpose_0_op_output0", "grid0_reshape_0_op_shape"],
        outputs=["grid0_reshape_0_op_output0"])

    #### Slice操作和Sigmoid操作得到xy
    grid0_slice_0_op_start = make_tensor_value_info("grid0_slice_0_op_start", TensorProto.INT64, [1])
    grid0_slice_0_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64))
    grid0_slice_0_op_start_init.name = "grid0_slice_0_op_start"

    grid0_slice_0_op_end = make_tensor_value_info("grid0_slice_0_op_end", TensorProto.INT64, [1])
    grid0_slice_0_op_end_init = numpy_helper.from_array(np.array([19, 19, 3, 2], dtype=np.int64))
    grid0_slice_0_op_end_init.name = "grid0_slice_0_op_end"

    grid0_slice_0_op_axis = make_tensor_value_info("grid0_slice_0_op_axis", TensorProto.INT64, [1])
    grid0_slice_0_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid0_slice_0_op_axis_init.name = "grid0_slice_0_op_axis"

    grid0_slice_0_op_output0 = make_tensor_value_info("grid0_slice_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_slice_0_op = make_node(
        "Slice",
        inputs=["grid0_reshape_0_op_output0", "grid0_slice_0_op_start", "grid0_slice_0_op_end", "grid0_slice_0_op_axis"],
        outputs=["grid0_slice_0_op_output0"])

    grid0_sigmoid_0_op_output0 = make_tensor_value_info("grid0_sigmoid_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_sigmoid_0_op = make_node(
        "Sigmoid",
        inputs=["grid0_slice_0_op_output0"],
        outputs=["grid0_sigmoid_0_op_output0"])

    #### Slice操作和Sigmoid操作得到class_probs
    grid0_slice_1_op_start = make_tensor_value_info("grid0_slice_1_op_start", TensorProto.INT64, [1])
    grid0_slice_1_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 5], dtype=np.int64))
    grid0_slice_1_op_start_init.name = "grid0_slice_1_op_start"

    grid0_slice_1_op_end = make_tensor_value_info("grid0_slice_1_op_end", TensorProto.INT64, [1])
    grid0_slice_1_op_end_init = numpy_helper.from_array(np.array([19, 19, 3, 85], dtype=np.int64))
    grid0_slice_1_op_end_init.name = "grid0_slice_1_op_end"

    grid0_slice_1_op_axis = make_tensor_value_info("grid0_slice_1_op_axis", TensorProto.INT64, [1])
    grid0_slice_1_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid0_slice_1_op_axis_init.name = "grid0_slice_1_op_axis"

    grid0_slice_1_op_output0 = make_tensor_value_info("grid0_slice_1_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 80])
    grid0_slice_1_op = make_node(
        "Slice",
        inputs=["grid0_reshape_0_op_output0", "grid0_slice_1_op_start", "grid0_slice_1_op_end", "grid0_slice_1_op_axis"],
        outputs=["grid0_slice_1_op_output0"])

    grid0_sigmoid_1_op_output0 = make_tensor_value_info("grid0_sigmoid_1_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 80])
    grid0_sigmoid_1_op = make_node(
        "Sigmoid",
        inputs=["grid0_slice_1_op_output0"],
        outputs=["grid0_sigmoid_1_op_output0"])

    #### Slice操作和sigmoid得到class_confidence
    grid0_slice_2_op_start = make_tensor_value_info("grid0_slice_2_op_start", TensorProto.INT64, [1])
    grid0_slice_2_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 4], dtype=np.int64))
    grid0_slice_2_op_start_init.name = "grid0_slice_2_op_start"

    grid0_slice_2_op_end = make_tensor_value_info("grid0_slice_2_op_end", TensorProto.INT64, [1])
    grid0_slice_2_op_end_init = numpy_helper.from_array(np.array([19, 19, 3, 5], dtype=np.int64))
    grid0_slice_2_op_end_init.name = "grid0_slice_2_op_end"

    grid0_slice_2_op_axis = make_tensor_value_info("grid0_slice_2_op_axis", TensorProto.INT64, [1])
    grid0_slice_2_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid0_slice_2_op_axis_init.name = "grid0_slice_2_op_axis"

    grid0_slice_2_op_output0 = make_tensor_value_info("grid0_slice_2_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 1])
    grid0_slice_2_op = make_node(
        "Slice",
        inputs=["grid0_reshape_0_op_output0", "grid0_slice_2_op_start", "grid0_slice_2_op_end", "grid0_slice_2_op_axis"],
        outputs=["grid0_slice_2_op_output0"])

    grid0_sigmoid_2_op_output0 = make_tensor_value_info("grid0_sigmoid_2_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 1])
    grid0_sigmoid_2_op = make_node(
        "Sigmoid",
        inputs=["grid0_slice_2_op_output0"],
        outputs=["grid0_sigmoid_2_op_output0"])

    #### Slice操作, Exp操作还有Mul操作得到wh
    grid0_slice_3_op_start = make_tensor_value_info("grid0_slice_3_op_start", TensorProto.INT64, [1])
    grid0_slice_3_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 2], dtype=np.int64))
    grid0_slice_3_op_start_init.name = "grid0_slice_3_op_start"

    grid0_slice_3_op_end = make_tensor_value_info("grid0_slice_3_op_end", TensorProto.INT64, [1])
    grid0_slice_3_op_end_init = numpy_helper.from_array(np.array([19, 19, 3, 4], dtype=np.int64))
    grid0_slice_3_op_end_init.name = "grid0_slice_3_op_end"

    grid0_slice_3_op_axis = make_tensor_value_info("grid0_slice_3_op_axis", TensorProto.INT64, [1])
    grid0_slice_3_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid0_slice_3_op_axis_init.name = "grid0_slice_3_op_axis"

    grid0_slice_3_op_output0 = make_tensor_value_info("grid0_slice_3_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_slice_3_op = make_node(
        "Slice",
        inputs=["grid0_reshape_0_op_output0", "grid0_slice_3_op_start", "grid0_slice_3_op_end", "grid0_slice_3_op_axis"],
        outputs=["grid0_slice_3_op_output0"])

    grid0_exponent_0_op_output0 = make_tensor_value_info("grid0_exponent_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_exponent_0_op = make_node(
        "Exp",
        inputs=["grid0_slice_3_op_output0"],
        outputs=["grid0_exponent_0_op_output0"])

    # 初始化你的anchors
    grid0_anchors = make_tensor_value_info("grid0_anchors", TensorProto.INT64, [1])
    grid0_anchors_init = numpy_helper.from_array(np.array([[[[116, 90], [156, 198], [373, 326]]]], dtype=np.float32))
    grid0_anchors_init.name = "grid0_anchors"

    grid0_multi_0_op_output0 = make_tensor_value_info("grid0_multi_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_multi_0_op = make_node(
        "Mul",
        inputs=["grid0_exponent_0_op_output0", "grid0_anchors"],
        outputs=["grid0_multi_0_op_output0"])

    #### 构造grid0
    # col
    grid0_tile_0_op_input0 = make_tensor_value_info("grid0_tile_0_op_input0", TensorProto.INT64, [1])
    grid0_tile_0_op_input0_init = numpy_helper.from_array(np.arange(0, 19, dtype=np.float32))
    grid0_tile_0_op_input0_init.name = "grid0_tile_0_op_input0"

    grid0_tile_0_op_repeats = make_tensor_value_info("grid0_tile_0_op_repeats", TensorProto.INT64, [1])
    grid0_tile_0_op_repeats_init = numpy_helper.from_array(np.array([19], dtype=np.int64))
    grid0_tile_0_op_repeats_init.name = "grid0_tile_0_op_repeats"
    
    grid0_tile_0_op_output0 = make_tensor_value_info("grid0_tile_0_op_output0", TensorProto.FLOAT, [19*19,])
    grid0_tile_0_op = make_node(
        "Tile",
        inputs=["grid0_tile_0_op_input0", "grid0_tile_0_op_repeats"],
        outputs=["grid0_tile_0_op_output0"])

    grid0_reshape_1_op_shape = make_tensor_value_info("grid0_reshape_1_op_shape", TensorProto.INT64, [1])
    grid0_reshape_1_op_shape_init = numpy_helper.from_array(np.array([19, 19, 1, 1]))
    grid0_reshape_1_op_shape_init.name = "grid0_reshape_1_op_shape"

    grid0_reshape_1_op_output0 = make_tensor_value_info("grid0_reshape_1_op_output0", TensorProto.FLOAT, [19, 19, 1, 1])
    grid0_reshape_1_op = make_node(
        "Reshape",
        inputs=["grid0_tile_0_op_output0", "grid0_reshape_1_op_shape"],
        outputs=["grid0_reshape_1_op_output0"])

    grid0_concat_0_op_output0 = make_tensor_value_info("grid0_concat_0_op_output0", TensorProto.FLOAT, [19, 19, 3, 1])
    grid0_concat_0_op = make_node(
        "Concat",
        inputs=["grid0_reshape_1_op_output0", "grid0_reshape_1_op_output0", "grid0_reshape_1_op_output0"],
        outputs=["grid0_concat_0_op_output0"],
        axis=-2)

    # row
    grid0_tile_1_op_input0 = make_tensor_value_info("grid0_tile_1_op_input0", TensorProto.INT64, [1])
    grid0_tile_1_op_input0_init = numpy_helper.from_array(np.arange(0, 19, dtype=np.float32).reshape(-1, 1))
    grid0_tile_1_op_input0_init.name = "grid0_tile_1_op_input0"

    grid0_tile_1_op_repeats = make_tensor_value_info("grid0_tile_1_op_repeats", TensorProto.INT64, [1])
    grid0_tile_1_op_repeats_init = numpy_helper.from_array(np.array([1, 19], dtype=np.int64))
    grid0_tile_1_op_repeats_init.name = "grid0_tile_1_op_repeats"

    grid0_tile_1_op_output0 = make_tensor_value_info("grid0_tile_1_op_output0", TensorProto.FLOAT, [19, 19])
    grid0_tile_1_op = make_node(
        "Tile",
        inputs=["grid0_tile_1_op_input0", "grid0_tile_1_op_repeats"],
        outputs=["grid0_tile_1_op_output0"])

    grid0_reshape_2_op_shape = make_tensor_value_info("grid0_reshape_2_op_shape", TensorProto.INT64, [1])
    grid0_reshape_2_op_shape_init = numpy_helper.from_array(np.array([19, 19, 1, 1]))
    grid0_reshape_2_op_shape_init.name = "grid0_reshape_2_op_shape"

    grid0_reshape_2_op_output0 = make_tensor_value_info("grid0_reshape_2_op_output0", TensorProto.FLOAT, [19, 19, 1, 1])
    grid0_reshape_2_op = make_node(
        "Reshape",
        inputs=["grid0_tile_1_op_output0", "grid0_reshape_2_op_shape"],
        outputs=["grid0_reshape_2_op_output0"])

    grid0_concat_1_op_output0 = make_tensor_value_info("grid0_concat_1_op_output0", TensorProto.FLOAT, [19, 19, 3, 1])
    grid0_concat_1_op = make_node(
        "Concat",
        inputs=["grid0_reshape_2_op_output0", "grid0_reshape_2_op_output0", "grid0_reshape_2_op_output0"],
        outputs=["grid0_concat_1_op_output0"],
        axis=-2)

    # grid
    grid0_concat_2_op_output0 = make_tensor_value_info("grid0_concat_2_op_output0", TensorProto.FLOAT, [19, 19, 3, 2])
    grid0_concat_2_op = make_node(
        "Concat",
        inputs=["grid0_concat_0_op_output0", "grid0_concat_1_op_output0"],
        outputs=["grid0_concat_2_op_output0"],
        axis=-1)

    #### 构造新的box_xy, box_wh
    grid0_add_0_op_output0 = make_tensor_value_info("grid0_add_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_add_0_op = make_node(
        "Add",
        inputs=["grid0_concat_2_op_output0", "grid0_sigmoid_0_op_output0"],
        outputs=["grid0_add_0_op_output0"])

    grid0_div_0_op_input1 = make_tensor_value_info("grid0_div_0_op_input1", TensorProto.INT64, [1])
    grid0_div_0_op_input1_init = numpy_helper.from_array(np.array([19., 19.], dtype=np.float32))
    grid0_div_0_op_input1_init.name = "grid0_div_0_op_input1"

    grid0_div_0_op_output0 = make_tensor_value_info("grid0_div_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_div_0_op = make_node(
        "Div",
        inputs=["grid0_add_0_op_output0", "grid0_div_0_op_input1"],
        outputs=["grid0_div_0_op_output0"])

    inputresolution_tuple = make_tensor_value_info("inputresolution_tuple", TensorProto.INT64, [1])
    inputresolution_tuple_init = numpy_helper.from_array(np.array([608., 608.], dtype=np.float32))
    inputresolution_tuple_init.name = "inputresolution_tuple"

    grid0_div_1_op_output0 = make_tensor_value_info("grid0_div_1_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_div_1_op = make_node(
        "Div",
        inputs=["grid0_multi_0_op_output0", "inputresolution_tuple"],
        outputs=["grid0_div_1_op_output0"])

    two = two = make_tensor_value_info("two", TensorProto.INT64, [1])
    two_init = numpy_helper.from_array(np.array(2, dtype=np.float32))
    two_init.name = "two"

    grid0_div_2_op_output0 = make_tensor_value_info("grid0_div_2_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_div_2_op = make_node(
        "Div",
        inputs=["grid0_div_1_op_output0", "two"],
        outputs=["grid0_div_2_op_output0"])

    grid0_sub_0_op_output0 = make_tensor_value_info("grid0_sub_0_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 2])
    grid0_sub_0_op = make_node(
        "Sub",
        inputs=["grid0_div_0_op_output0", "grid0_div_2_op_output0"],
        outputs=["grid0_sub_0_op_output0"])

    grid0_concat_3_op_output0 = make_tensor_value_info("grid0_concat_3_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 4])
    grid0_concat_3_op = make_node(
        "Concat",
        inputs=["grid0_sub_0_op_output0", "grid0_div_1_op_output0"],
        outputs=["grid0_concat_3_op_output0"],
        axis=-1)

    #### 接下来写filter_boxes函数, 输入是: grid0_concat_3_op_output0, grid0_sigmoid_2_op_output0, grid0_sigmoid_1_op_output0
    ## grid0_concat_3_op_output0:  [-1, 19, 19, 3,  4]  --->  boxes
    ## grid0_sigmoid_2_op_output0: [-1, 19, 19, 3,  1]  --->  box_confidences
    ## grid0_sigmoid_1_op_output0: [-1, 19, 19, 3, 80]  --->  box_class_probs

    grid0_reshape_3_op_shape = make_tensor_value_info("grid0_reshape_3_op_shape", TensorProto.INT64, [1])
    grid0_reshape_3_op_shape_init = numpy_helper.from_array(np.array([-1, 19*19*3, 4]))
    grid0_reshape_3_op_shape_init.name = "grid0_reshape_3_op_shape"

    grid0_reshape_3_op_output0 = make_tensor_value_info("grid0_reshape_3_op_output0", TensorProto.FLOAT, ["-1", 19*19*3, 4])
    grid0_reshape_3_op = make_node(
        "Reshape",
        inputs=["grid0_concat_3_op_output0", "grid0_reshape_3_op_shape"],
        outputs=["grid0_reshape_3_op_output0"])

    grid0_positions = make_tensor_value_info("grid0_positions", TensorProto.FLOAT, ["-1", 19*19*3, 4]) # grid0的最终坐标
    grid0_multi_1_op = make_node( # 坐标
        "Mul",
        inputs=["grid0_reshape_3_op_output0", "origin_input_resolution"],
        outputs=["grid0_positions"])

    grid0_multi_2_op_output0 = make_tensor_value_info("grid0_multi_2_op_output0", TensorProto.FLOAT, ["-1", 19, 19, 3, 80])
    grid0_multi_2_op = make_node( # 分数
        "Mul",
        inputs=["grid0_sigmoid_2_op_output0", "grid0_sigmoid_1_op_output0"],
        outputs=["grid0_multi_2_op_output0"])

    grid0_reshape_4_op_shape = make_tensor_value_info("grid0_reshape_4_op_shape", TensorProto.INT64, [1])
    grid0_reshape_4_op_shape_init = numpy_helper.from_array(np.array([-1, 19*19*3, 80], dtype=np.int64))
    grid0_reshape_4_op_shape_init.name = "grid0_reshape_4_op_shape"

    grid0_scores = make_tensor_value_info("grid0_scores", TensorProto.FLOAT, ["-1", 19*19*3, 80])
    grid0_reshape_4_op = make_node( # 分数reshape一下, grid0的最终分数
        "Reshape",
        inputs=["grid0_multi_2_op_output0", "grid0_reshape_4_op_shape"],
        outputs=["grid0_scores"])

    ########## Grid1: -1x255x38x38 ##########
    grid1_transpose_0_op_output0 = make_tensor_value_info("grid1_transpose_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 255])
    grid1_transpose_0_op = make_node(
        "Transpose",
        perm=[0, 2, 3, 1],
        inputs=["output1"],
        outputs=["grid1_transpose_0_op_output0"])

    grid1_reshape_0_op_shape = make_tensor_value_info("grid1_reshape_0_op_shape", TensorProto.INT64, [1])
    grid1_reshape_0_op_shape_init = numpy_helper.from_array(np.array([-1, 38, 38, 3, 85], dtype=np.int64))
    grid1_reshape_0_op_shape_init.name = "grid1_reshape_0_op_shape"

    grid1_reshape_0_op_output0 = make_tensor_value_info("grid1_reshape_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 85])
    grid1_reshape_0_op = make_node(
        "Reshape",
        inputs=["grid1_transpose_0_op_output0", "grid1_reshape_0_op_shape"],
        outputs=["grid1_reshape_0_op_output0"])

    #### Slice操作和Sigmoid操作得到xy
    grid1_slice_0_op_start = make_tensor_value_info("grid1_slice_0_op_start", TensorProto.INT64, [1])
    grid1_slice_0_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64))
    grid1_slice_0_op_start_init.name = "grid1_slice_0_op_start"

    grid1_slice_0_op_end = make_tensor_value_info("grid1_slice_0_op_end", TensorProto.INT64, [1])
    grid1_slice_0_op_end_init = numpy_helper.from_array(np.array([38, 38, 3, 2], dtype=np.int64))
    grid1_slice_0_op_end_init.name = "grid1_slice_0_op_end"

    grid1_slice_0_op_axis = make_tensor_value_info("grid1_slice_0_op_axis", TensorProto.INT64, [1])
    grid1_slice_0_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid1_slice_0_op_axis_init.name = "grid1_slice_0_op_axis"

    grid1_slice_0_op_output0 = make_tensor_value_info("grid1_slice_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_slice_0_op = make_node(
        "Slice",
        inputs=["grid1_reshape_0_op_output0", "grid1_slice_0_op_start", "grid1_slice_0_op_end", "grid1_slice_0_op_axis"],
        outputs=["grid1_slice_0_op_output0"])

    grid1_sigmoid_0_op_output0 = make_tensor_value_info("grid1_sigmoid_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_sigmoid_0_op = make_node(
        "Sigmoid",
        inputs=["grid1_slice_0_op_output0"],
        outputs=["grid1_sigmoid_0_op_output0"])

    #### Slice操作和Sigmoid操作得到class_probs
    grid1_slice_1_op_start = make_tensor_value_info("grid1_slice_1_op_start", TensorProto.INT64, [1])
    grid1_slice_1_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 5], dtype=np.int64))
    grid1_slice_1_op_start_init.name = "grid1_slice_1_op_start"

    grid1_slice_1_op_end = make_tensor_value_info("grid1_slice_1_op_end", TensorProto.INT64, [1])
    grid1_slice_1_op_end_init = numpy_helper.from_array(np.array([38, 38, 3, 85], dtype=np.int64))
    grid1_slice_1_op_end_init.name = "grid1_slice_1_op_end"

    grid1_slice_1_op_axis = make_tensor_value_info("grid1_slice_1_op_axis", TensorProto.INT64, [1])
    grid1_slice_1_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid1_slice_1_op_axis_init.name = "grid1_slice_1_op_axis"

    grid1_slice_1_op_output0 = make_tensor_value_info("grid1_slice_1_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 80])
    grid1_slice_1_op = make_node(
        "Slice",
        inputs=["grid1_reshape_0_op_output0", "grid1_slice_1_op_start", "grid1_slice_1_op_end", "grid1_slice_1_op_axis"],
        outputs=["grid1_slice_1_op_output0"])

    grid1_sigmoid_1_op_output0 = make_tensor_value_info("grid1_sigmoid_1_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 80])
    grid1_sigmoid_1_op = make_node(
        "Sigmoid",
        inputs=["grid1_slice_1_op_output0"],
        outputs=["grid1_sigmoid_1_op_output0"])

    #### Slice操作和Sigmoid得到class_confidence
    grid1_slice_2_op_start = make_tensor_value_info("grid1_slice_2_op_start", TensorProto.INT64, [1])
    grid1_slice_2_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 4], dtype=np.int64))
    grid1_slice_2_op_start_init.name = "grid1_slice_2_op_start"

    grid1_slice_2_op_end = make_tensor_value_info("grid1_slice_2_op_end", TensorProto.INT64, [1])
    grid1_slice_2_op_end_init = numpy_helper.from_array(np.array([38, 38, 3, 5], dtype=np.int64))
    grid1_slice_2_op_end_init.name = "grid1_slice_2_op_end"

    grid1_slice_2_op_axis = make_tensor_value_info("grid1_slice_2_op_axis", TensorProto.INT64, [1])
    grid1_slice_2_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid1_slice_2_op_axis_init.name = "grid1_slice_2_op_axis"

    grid1_slice_2_op_output0 = make_tensor_value_info("grid1_slice_2_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 1])
    grid1_slice_2_op = make_node(
        "Slice",
        inputs=["grid1_reshape_0_op_output0", "grid1_slice_2_op_start", "grid1_slice_2_op_end", "grid1_slice_2_op_axis"],
        outputs=["grid1_slice_2_op_output0"])

    grid1_sigmoid_2_op_output0 = make_tensor_value_info("grid1_sigmoid_2_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 1])
    grid1_sigmoid_2_op = make_node(
        "Sigmoid",
        inputs=["grid1_slice_2_op_output0"],
        outputs=["grid1_sigmoid_2_op_output0"])

    #### Slice操作, Exp操作还有Mul操作得到wh
    grid1_slice_3_op_start = make_tensor_value_info("grid1_slice_3_op_start", TensorProto.INT64, [1])
    grid1_slice_3_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 2], dtype=np.int64))
    grid1_slice_3_op_start_init.name = "grid1_slice_3_op_start"

    grid1_slice_3_op_end = make_tensor_value_info("grid1_slice_3_op_end", TensorProto.INT64, [1])
    grid1_slice_3_op_end_init = numpy_helper.from_array(np.array([38, 38, 3, 4], dtype=np.int64))
    grid1_slice_3_op_end_init.name = "grid1_slice_3_op_end"

    grid1_slice_3_op_axis = make_tensor_value_info("grid1_slice_3_op_axis", TensorProto.INT64, [1])
    grid1_slice_3_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid1_slice_3_op_axis_init.name = "grid1_slice_3_op_axis"

    grid1_slice_3_op_output0 = make_tensor_value_info("grid1_slice_3_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_slice_3_op = make_node(
        "Slice",
        inputs=["grid1_reshape_0_op_output0", "grid1_slice_3_op_start", "grid1_slice_3_op_end", "grid1_slice_3_op_axis"],
        outputs=["grid1_slice_3_op_output0"])

    grid1_exponent_0_op_output0 = make_tensor_value_info("grid1_exponent_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_exponent_0_op = make_node(
        "Exp",
        inputs=["grid1_slice_3_op_output0"],
        outputs=["grid1_exponent_0_op_output0"])

    # 初始化你的anchors
    grid1_anchors = make_tensor_value_info("grid1_anchors", TensorProto.INT64, [1])
    grid1_anchors_init = numpy_helper.from_array(np.array([[[[30, 61], [62, 45], [59, 119]]]], dtype=np.float32))
    grid1_anchors_init.name = "grid1_anchors"

    grid1_multi_0_op_output0 = make_tensor_value_info("grid1_multi_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_multi_0_op = make_node(
        "Mul",
        inputs=["grid1_exponent_0_op_output0", "grid1_anchors"],
        outputs=["grid1_multi_0_op_output0"])
    
    #### 构造grid1
    # col
    grid1_tile_0_op_input0 = make_tensor_value_info("grid1_tile_0_op_input0", TensorProto.INT64, [1])
    grid1_tile_0_op_input0_init = numpy_helper.from_array(np.arange(0, 38, dtype=np.float32))
    grid1_tile_0_op_input0_init.name = "grid1_tile_0_op_input0"

    grid1_tile_0_op_repeats = make_tensor_value_info("grid1_tile_0_op_repeats", TensorProto.INT64, [1])
    grid1_tile_0_op_repeats_init = numpy_helper.from_array(np.array([38], dtype=np.int64))
    grid1_tile_0_op_repeats_init.name = "grid1_tile_0_op_repeats"

    grid1_tile_0_op_output0 = make_tensor_value_info("grid1_tile_0_op_output0", TensorProto.FLOAT, [38*38,])
    grid1_tile_0_op = make_node(
        "Tile",
        inputs=["grid1_tile_0_op_input0", "grid1_tile_0_op_repeats"],
        outputs=["grid1_tile_0_op_output0"])

    grid1_reshape_1_op_shape = make_tensor_value_info("grid1_reshape_1_op_shape", TensorProto.INT64, [1])
    grid1_reshape_1_op_shape_init = numpy_helper.from_array(np.array([38, 38, 1, 1]))
    grid1_reshape_1_op_shape_init.name = "grid1_reshape_1_op_shape"

    grid1_reshape_1_op_output0 = make_tensor_value_info("grid1_reshape_1_op_output0", TensorProto.FLOAT, [38, 38, 1, 1])
    grid1_reshape_1_op = make_node(
        "Reshape",
        inputs=["grid1_tile_0_op_output0", "grid1_reshape_1_op_shape"],
        outputs=["grid1_reshape_1_op_output0"])

    grid1_concat_0_op_output0 = make_tensor_value_info("grid1_concat_0_op_output0", TensorProto.FLOAT, [38, 38, 3, 1])
    grid1_concat_0_op = make_node(
        "Concat",
        inputs=["grid1_reshape_1_op_output0", "grid1_reshape_1_op_output0", "grid1_reshape_1_op_output0"],
        outputs=["grid1_concat_0_op_output0"],
        axis=-2)

    # row
    grid1_tile_1_op_input0 = make_tensor_value_info("grid1_tile_1_op_input0", TensorProto.INT64, [1])
    grid1_tile_1_op_input0_init = numpy_helper.from_array(np.arange(0, 38, dtype=np.float32).reshape(-1, 1))
    grid1_tile_1_op_input0_init.name = "grid1_tile_1_op_input0"

    grid1_tile_1_op_repeats = make_tensor_value_info("grid1_tile_1_op_repeats", TensorProto.INT64, [1])
    grid1_tile_1_op_repeats_init = numpy_helper.from_array(np.array([1, 38], dtype=np.int64))
    grid1_tile_1_op_repeats_init.name = "grid1_tile_1_op_repeats"

    grid1_tile_1_op_output0 = make_tensor_value_info("grid1_tile_1_op_output0", TensorProto.FLOAT, [38, 38])
    grid1_tile_1_op = make_node(
        "Tile",
        inputs=["grid1_tile_1_op_input0", "grid1_tile_1_op_repeats"],
        outputs=["grid1_tile_1_op_output0"])

    grid1_reshape_2_op_shape = make_tensor_value_info("grid1_reshape_2_op_shape", TensorProto.INT64, [1])
    grid1_reshape_2_op_shape_init = numpy_helper.from_array(np.array([38, 38, 1, 1]))
    grid1_reshape_2_op_shape_init.name = "grid1_reshape_2_op_shape"

    grid1_reshape_2_op_output0 = make_tensor_value_info("grid1_reshape_2_op_output0", TensorProto.FLOAT, [38, 38, 1, 1])
    grid1_reshape_2_op = make_node(
        "Reshape",
        inputs=["grid1_tile_1_op_output0", "grid1_reshape_2_op_shape"],
        outputs=["grid1_reshape_2_op_output0"])

    grid1_concat_1_op_output0 = make_tensor_value_info("grid1_concat_1_op_output0", TensorProto.FLOAT, [38, 38, 3, 1])
    grid1_concat_1_op = make_node(
        "Concat",
        inputs=["grid1_reshape_2_op_output0", "grid1_reshape_2_op_output0", "grid1_reshape_2_op_output0"],
        outputs=["grid1_concat_1_op_output0"],
        axis=-2)

    # grid
    grid1_concat_2_op_output0 = make_tensor_value_info("grid1_concat_2_op_output0", TensorProto.FLOAT, [38, 38, 3, 2])
    grid1_concat_2_op = make_node(
        "Concat",
        inputs=["grid1_concat_0_op_output0", "grid1_concat_1_op_output0"],
        outputs=["grid1_concat_2_op_output0"],
        axis=-1)

    #### 构造新的box_xy, box_wh
    grid1_add_0_op_output0 = make_tensor_value_info("grid1_add_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_add_0_op = make_node(
        "Add",
        inputs=["grid1_concat_2_op_output0", "grid1_sigmoid_0_op_output0"],
        outputs=["grid1_add_0_op_output0"])

    grid1_div_0_op_input1 = make_tensor_value_info("grid1_div_0_op_input1", TensorProto.INT64, [1])
    grid1_div_0_op_input1_init = numpy_helper.from_array(np.array([38., 38.], dtype=np.float32))
    grid1_div_0_op_input1_init.name = "grid1_div_0_op_input1"

    grid1_div_0_op_output0 = make_tensor_value_info("grid1_div_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_div_0_op = make_node(
        "Div",
        inputs=["grid1_add_0_op_output0", "grid1_div_0_op_input1"],
        outputs=["grid1_div_0_op_output0"])

    grid1_div_1_op_output0 = make_tensor_value_info("grid1_div_1_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_div_1_op = make_node(
        "Div",
        inputs=["grid1_multi_0_op_output0", "inputresolution_tuple"],
        outputs=["grid1_div_1_op_output0"])

    grid1_div_2_op_output0 = make_tensor_value_info("grid1_div_2_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_div_2_op = make_node(
        "Div",
        inputs=["grid1_div_1_op_output0", "two"],
        outputs=["grid1_div_2_op_output0"])

    grid1_sub_0_op_output0 = make_tensor_value_info("grid1_sub_0_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 2])
    grid1_sub_0_op = make_node(
        "Sub",
        inputs=["grid1_div_0_op_output0", "grid1_div_2_op_output0"],
        outputs=["grid1_sub_0_op_output0"])

    grid1_concat_3_op_output0 = make_tensor_value_info("grid1_concat_3_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 4])
    grid1_concat_3_op = make_node(
        "Concat",
        inputs=["grid1_sub_0_op_output0", "grid1_div_1_op_output0"],
        outputs=["grid1_concat_3_op_output0"],
        axis=-1)

    #### 接下来写filter_boxes函数, 输入是: grid1_concat_3_op_output0, grid1_sigmoid_2_op_output0, grid1_sigmoid_1_op_output0
    ## grid1_concat_3_op_output0:  [-1, 38, 38, 3,  4]  --->  boxes
    ## grid1_sigmoid_2_op_output0: [-1, 38, 38, 3,  1]  --->  box_confidences
    ## grid1_sigmoid_1_op_output0: [-1, 38, 38, 3, 80]  --->  box_class_probs
    grid1_reshape_3_op_shape = make_tensor_value_info("grid1_reshape_3_op_shape", TensorProto.INT64, [1])
    grid1_reshape_3_op_shape_init = numpy_helper.from_array(np.array([-1, 38*38*3, 4]))
    grid1_reshape_3_op_shape_init.name = "grid1_reshape_3_op_shape"

    grid1_reshape_3_op_output0 = make_tensor_value_info("grid1_reshape_3_op_output0", TensorProto.FLOAT, ["-1", 38*38*3, 4])
    grid1_reshape_3_op = make_node(
        "Reshape",
        inputs=["grid1_concat_3_op_output0", "grid1_reshape_3_op_shape"],
        outputs=["grid1_reshape_3_op_output0"])

    grid1_positions = make_tensor_value_info("grid1_positions", TensorProto.FLOAT, ["-1", 38*38*3, 4]) # grid1的最终坐标
    grid1_multi_1_op = make_node( # 坐标
        "Mul",
        inputs=["grid1_reshape_3_op_output0", "origin_input_resolution"],
        outputs=["grid1_positions"])

    grid1_multi_2_op_output0 = make_tensor_value_info("grid1_multi_2_op_output0", TensorProto.FLOAT, ["-1", 38, 38, 3, 80])
    grid1_multi_2_op = make_node( # 分数
        "Mul",
        inputs=["grid1_sigmoid_2_op_output0", "grid1_sigmoid_1_op_output0"],
        outputs=["grid1_multi_2_op_output0"])

    grid1_reshape_4_op_shape = make_tensor_value_info("grid1_reshape_4_op_shape", TensorProto.INT64, [1])
    grid1_reshape_4_op_shape_init = numpy_helper.from_array(np.array([-1, 38*38*3, 80], dtype=np.int64))
    grid1_reshape_4_op_shape_init.name = "grid1_reshape_4_op_shape"

    grid1_scores = make_tensor_value_info("grid1_scores", TensorProto.FLOAT, ["-1", 38*38*3, 80])
    grid1_reshape_4_op = make_node( # 分数reshape一下, grid1的最终分数
        "Reshape",
        inputs=["grid1_multi_2_op_output0", "grid1_reshape_4_op_shape"],
        outputs=["grid1_scores"])
    
    ########## Grid2: -1x255x76x76 ##########
    grid2_transpose_0_op_output0 = make_tensor_value_info("grid2_transpose_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 255])
    grid2_transpose_0_op = make_node(
        "Transpose",
        perm=[0, 2, 3, 1],
        inputs=["output2"],
        outputs=["grid2_transpose_0_op_output0"])

    grid2_reshape_0_op_shape = make_tensor_value_info("grid2_reshape_0_op_shape", TensorProto.INT64, [1])
    grid2_reshape_0_op_shape_init = numpy_helper.from_array(np.array([-1, 76, 76, 3, 85], dtype=np.int64))
    grid2_reshape_0_op_shape_init.name = "grid2_reshape_0_op_shape"

    grid2_reshape_0_op_output0 = make_tensor_value_info("grid2_reshape_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 85])
    grid2_reshape_0_op = make_node(
        "Reshape",
        inputs=["grid2_transpose_0_op_output0", "grid2_reshape_0_op_shape"],
        outputs=["grid2_reshape_0_op_output0"])

    #### Slice操作和Sigmoid操作得到xy
    grid2_slice_0_op_start = make_tensor_value_info("grid2_slice_0_op_start", TensorProto.INT64, [1])
    grid2_slice_0_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 0], dtype=np.int64))
    grid2_slice_0_op_start_init.name = "grid2_slice_0_op_start"

    grid2_slice_0_op_end = make_tensor_value_info("grid2_slice_0_op_end", TensorProto.INT64, [1])
    grid2_slice_0_op_end_init = numpy_helper.from_array(np.array([76, 76, 3, 2], dtype=np.int64))
    grid2_slice_0_op_end_init.name = "grid2_slice_0_op_end"

    grid2_slice_0_op_axis = make_tensor_value_info("grid2_slice_0_op_axis", TensorProto.INT64, [1])
    grid2_slice_0_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid2_slice_0_op_axis_init.name = "grid2_slice_0_op_axis"

    grid2_slice_0_op_output0 = make_tensor_value_info("grid2_slice_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_slice_0_op = make_node(
        "Slice",
        inputs=["grid2_reshape_0_op_output0", "grid2_slice_0_op_start", "grid2_slice_0_op_end", "grid2_slice_0_op_axis"],
        outputs=["grid2_slice_0_op_output0"])

    grid2_sigmoid_0_op_output0 = make_tensor_value_info("grid2_sigmoid_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_sigmoid_0_op = make_node(
        "Sigmoid",
        inputs=["grid2_slice_0_op_output0"],
        outputs=["grid2_sigmoid_0_op_output0"])

    #### Slice操作和Sigmoid操作得到class_probs
    grid2_slice_1_op_start = make_tensor_value_info("grid2_slice_1_op_start", TensorProto.INT64, [1])
    grid2_slice_1_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 5], dtype=np.int64))
    grid2_slice_1_op_start_init.name = "grid2_slice_1_op_start"

    grid2_slice_1_op_end = make_tensor_value_info("grid2_slice_1_op_end", TensorProto.INT64, [1])
    grid2_slice_1_op_end_init = numpy_helper.from_array(np.array([76, 76, 3, 85], dtype=np.int64))
    grid2_slice_1_op_end_init.name = "grid2_slice_1_op_end"

    grid2_slice_1_op_axis = make_tensor_value_info("grid2_slice_1_op_axis", TensorProto.INT64, [1])
    grid2_slice_1_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid2_slice_1_op_axis_init.name = "grid2_slice_1_op_axis"

    grid2_slice_1_op_output0 = make_tensor_value_info("grid2_slice_1_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 80])
    grid2_slice_1_op = make_node(
        "Slice",
        inputs=["grid2_reshape_0_op_output0", "grid2_slice_1_op_start", "grid2_slice_1_op_end", "grid2_slice_1_op_axis"],
        outputs=["grid2_slice_1_op_output0"])

    grid2_sigmoid_1_op_output0 = make_tensor_value_info("grid2_sigmoid_1_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 80])
    grid2_sigmoid_1_op = make_node(
        "Sigmoid",
        inputs=["grid2_slice_1_op_output0"],
        outputs=["grid2_sigmoid_1_op_output0"])

    #### Slice操作和Sigmoid得到class_confidence
    grid2_slice_2_op_start = make_tensor_value_info("grid2_slice_2_op_start", TensorProto.INT64, [1])
    grid2_slice_2_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 4], dtype=np.int64))
    grid2_slice_2_op_start_init.name = "grid2_slice_2_op_start"

    grid2_slice_2_op_end = make_tensor_value_info("grid2_slice_2_op_end", TensorProto.INT64, [1])
    grid2_slice_2_op_end_init = numpy_helper.from_array(np.array([76, 76, 3, 5], dtype=np.int64))
    grid2_slice_2_op_end_init.name = "grid2_slice_2_op_end"

    grid2_slice_2_op_axis = make_tensor_value_info("grid2_slice_2_op_axis", TensorProto.INT64, [1])
    grid2_slice_2_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid2_slice_2_op_axis_init.name = "grid2_slice_2_op_axis"

    grid2_slice_2_op_output0 = make_tensor_value_info("grid2_slice_2_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 1])
    grid2_slice_2_op = make_node(
        "Slice",
        inputs=["grid2_reshape_0_op_output0", "grid2_slice_2_op_start", "grid2_slice_2_op_end", "grid2_slice_2_op_axis"],
        outputs=["grid2_slice_2_op_output0"])

    grid2_sigmoid_2_op_output0 = make_tensor_value_info("grid2_sigmoid_2_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 1])
    grid2_sigmoid_2_op = make_node(
        "Sigmoid",
        inputs=["grid2_slice_2_op_output0"],
        outputs=["grid2_sigmoid_2_op_output0"])

    #### Slice操作, Exp操作还有Mul操作得到wh
    grid2_slice_3_op_start = make_tensor_value_info("grid2_slice_3_op_start", TensorProto.INT64, [1])
    grid2_slice_3_op_start_init = numpy_helper.from_array(np.array([0, 0, 0, 2], dtype=np.int64))
    grid2_slice_3_op_start_init.name = "grid2_slice_3_op_start"

    grid2_slice_3_op_end = make_tensor_value_info("grid2_slice_3_op_end", TensorProto.INT64, [1])
    grid2_slice_3_op_end_init = numpy_helper.from_array(np.array([76, 76, 3, 4], dtype=np.int64))
    grid2_slice_3_op_end_init.name = "grid2_slice_3_op_end"

    grid2_slice_3_op_axis = make_tensor_value_info("grid2_slice_3_op_axis", TensorProto.INT64, [1])
    grid2_slice_3_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3, 4], dtype=np.int64))
    grid2_slice_3_op_axis_init.name = "grid2_slice_3_op_axis"

    grid2_slice_3_op_output0 = make_tensor_value_info("grid2_slice_3_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_slice_3_op = make_node(
        "Slice",
        inputs=["grid2_reshape_0_op_output0", "grid2_slice_3_op_start", "grid2_slice_3_op_end", "grid2_slice_3_op_axis"],
        outputs=["grid2_slice_3_op_output0"])

    grid2_exponent_0_op_output0 = make_tensor_value_info("grid2_exponent_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_exponent_0_op = make_node(
        "Exp",
        inputs=["grid2_slice_3_op_output0"],
        outputs=["grid2_exponent_0_op_output0"])

    # 初始化你的anchors
    grid2_anchors = make_tensor_value_info("grid2_anchors", TensorProto.INT64, [1])
    grid2_anchors_init = numpy_helper.from_array(np.array([[[[10, 13], [16, 30], [33, 23]]]], dtype=np.float32))
    grid2_anchors_init.name = "grid2_anchors"

    grid2_multi_0_op_output0 = make_tensor_value_info("grid2_multi_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_multi_0_op = make_node(
        "Mul",
        inputs=["grid2_exponent_0_op_output0", "grid2_anchors"],
        outputs=["grid2_multi_0_op_output0"])

    #### 构造grid2
    # col
    grid2_tile_0_op_input0 = make_tensor_value_info("grid2_tile_0_op_input0", TensorProto.INT64, [1])
    grid2_tile_0_op_input0_init = numpy_helper.from_array(np.arange(0, 76, dtype=np.float32))
    grid2_tile_0_op_input0_init.name = "grid2_tile_0_op_input0"

    grid2_tile_0_op_repeats = make_tensor_value_info("grid2_tile_0_op_repeats", TensorProto.INT64, [1])
    grid2_tile_0_op_repeats_init = numpy_helper.from_array(np.array([76], dtype=np.int64))
    grid2_tile_0_op_repeats_init.name = "grid2_tile_0_op_repeats"

    grid2_tile_0_op_output0 = make_tensor_value_info("grid2_tile_0_op_output0", TensorProto.FLOAT, [76*76,])
    grid2_tile_0_op = make_node(
        "Tile",
        inputs=["grid2_tile_0_op_input0", "grid2_tile_0_op_repeats"],
        outputs=["grid2_tile_0_op_output0"])

    grid2_reshape_1_op_shape = make_tensor_value_info("grid2_reshape_1_op_shape", TensorProto.INT64, [1])
    grid2_reshape_1_op_shape_init = numpy_helper.from_array(np.array([76, 76, 1, 1]))
    grid2_reshape_1_op_shape_init.name = "grid2_reshape_1_op_shape"

    grid2_reshape_1_op_output0 = make_tensor_value_info("grid2_reshape_1_op_output0", TensorProto.FLOAT, [76, 76, 1, 1])
    grid2_reshape_1_op = make_node(
        "Reshape",
        inputs=["grid2_tile_0_op_output0", "grid2_reshape_1_op_shape"],
        outputs=["grid2_reshape_1_op_output0"])

    grid2_concat_0_op_output0 = make_tensor_value_info("grid2_concat_0_op_output0", TensorProto.FLOAT, [76, 76, 3, 1])
    grid2_concat_0_op = make_node(
        "Concat",
        inputs=["grid2_reshape_1_op_output0", "grid2_reshape_1_op_output0", "grid2_reshape_1_op_output0"],
        outputs=["grid2_concat_0_op_output0"],
        axis=-2)

    # row
    grid2_tile_1_op_input0 = make_tensor_value_info("grid2_tile_1_op_input0", TensorProto.INT64, [1])
    grid2_tile_1_op_input0_init = numpy_helper.from_array(np.arange(0, 76, dtype=np.float32).reshape(-1, 1))
    grid2_tile_1_op_input0_init.name = "grid2_tile_1_op_input0"

    grid2_tile_1_op_repeats = make_tensor_value_info("grid2_tile_1_op_repeats", TensorProto.INT64, [1])
    grid2_tile_1_op_repeats_init = numpy_helper.from_array(np.array([1, 76], dtype=np.int64))
    grid2_tile_1_op_repeats_init.name = "grid2_tile_1_op_repeats"

    grid2_tile_1_op_output0 = make_tensor_value_info("grid2_tile_1_op_output0", TensorProto.FLOAT, [76, 76])
    grid2_tile_1_op = make_node(
        "Tile",
        inputs=["grid2_tile_1_op_input0", "grid2_tile_1_op_repeats"],
        outputs=["grid2_tile_1_op_output0"])

    grid2_reshape_2_op_shape = make_tensor_value_info("grid2_reshape_2_op_shape", TensorProto.INT64, [1])
    grid2_reshape_2_op_shape_init = numpy_helper.from_array(np.array([76, 76, 1, 1]))
    grid2_reshape_2_op_shape_init.name = "grid2_reshape_2_op_shape"

    grid2_reshape_2_op_output0 = make_tensor_value_info("grid2_reshape_2_op_output0", TensorProto.FLOAT, [76, 76, 1, 1])
    grid2_reshape_2_op = make_node(
        "Reshape",
        inputs=["grid2_tile_1_op_output0", "grid2_reshape_2_op_shape"],
        outputs=["grid2_reshape_2_op_output0"])

    grid2_concat_1_op_output0 = make_tensor_value_info("grid2_concat_1_op_output0", TensorProto.FLOAT, [76, 76, 3, 1])
    grid2_concat_1_op = make_node(
        "Concat",
        inputs=["grid2_reshape_2_op_output0", "grid2_reshape_2_op_output0", "grid2_reshape_2_op_output0"],
        outputs=["grid2_concat_1_op_output0"],
        axis=-2)

    # grid
    grid2_concat_2_op_output0 = make_tensor_value_info("grid2_concat_2_op_output0", TensorProto.FLOAT, [76, 76, 3, 2])
    grid2_concat_2_op = make_node(
        "Concat",
        inputs=["grid2_concat_0_op_output0", "grid2_concat_1_op_output0"],
        outputs=["grid2_concat_2_op_output0"],
        axis=-1)

    #### 构造新的box_xy, box_wh
    grid2_add_0_op_output0 = make_tensor_value_info("grid2_add_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_add_0_op = make_node(
        "Add",
        inputs=["grid2_concat_2_op_output0", "grid2_sigmoid_0_op_output0"],
        outputs=["grid2_add_0_op_output0"])
    
    grid2_div_0_op_input1 = make_tensor_value_info("grid2_div_0_op_input1", TensorProto.INT64, [1])
    grid2_div_0_op_input1_init = numpy_helper.from_array(np.array([76., 76.], dtype=np.float32))
    grid2_div_0_op_input1_init.name = "grid2_div_0_op_input1"

    grid2_div_0_op_output0 = make_tensor_value_info("grid2_div_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_div_0_op = make_node(
        "Div",
        inputs=["grid2_add_0_op_output0", "grid2_div_0_op_input1"],
        outputs=["grid2_div_0_op_output0"])

    grid2_div_1_op_output0 = make_tensor_value_info("grid2_div_1_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_div_1_op = make_node(
        "Div",
        inputs=["grid2_multi_0_op_output0", "inputresolution_tuple"],
        outputs=["grid2_div_1_op_output0"])

    grid2_div_2_op_output0 = make_tensor_value_info("grid2_div_2_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_div_2_op = make_node(
        "Div",
        inputs=["grid2_div_1_op_output0", "two"],
        outputs=["grid2_div_2_op_output0"])

    grid2_sub_0_op_output0 = make_tensor_value_info("grid2_sub_0_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 2])
    grid2_sub_0_op = make_node(
        "Sub",
        inputs=["grid2_div_0_op_output0", "grid2_div_2_op_output0"],
        outputs=["grid2_sub_0_op_output0"])

    grid2_concat_3_op_output0 = make_tensor_value_info("grid2_concat_3_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 4])
    grid2_concat_3_op = make_node(
        "Concat",
        inputs=["grid2_sub_0_op_output0", "grid2_div_1_op_output0"],
        outputs=["grid2_concat_3_op_output0"],
        axis=-1)

    #### 接下来写filter_boxes函数, 输入是: grid2_concat_3_op_output0, grid2_sigmoid_2_op_output0, grid2_sigmoid_1_op_output0
    ## grid2_concat_3_op_output0:  [-1, 76, 76, 3,  4]  --->  boxes
    ## grid2_sigmoid_2_op_output0: [-1, 76, 76, 3,  1]  --->  box_confidences
    ## grid2_sigmoid_1_op_output0: [-1, 76, 76, 3, 80]  --->  box_class_probs
    grid2_reshape_3_op_shape = make_tensor_value_info("grid2_reshape_3_op_shape", TensorProto.INT64, [1])
    grid2_reshape_3_op_shape_init = numpy_helper.from_array(np.array([-1, 76*76*3, 4]))
    grid2_reshape_3_op_shape_init.name = "grid2_reshape_3_op_shape"

    grid2_reshape_3_op_output0 = make_tensor_value_info("grid2_reshape_3_op_output0", TensorProto.FLOAT, ["-1", 76*76*3, 4])
    grid2_reshape_3_op = make_node(
        "Reshape",
        inputs=["grid2_concat_3_op_output0", "grid2_reshape_3_op_shape"],
        outputs=["grid2_reshape_3_op_output0"])

    grid2_positions = make_tensor_value_info("grid2_positions", TensorProto.FLOAT, ["-1", 76*76*3, 4]) # grid2的最终坐标
    grid2_multi_1_op = make_node( # 坐标
        "Mul",
        inputs=["grid2_reshape_3_op_output0", "origin_input_resolution"],
        outputs=["grid2_positions"])

    grid2_multi_2_op_output0 = make_tensor_value_info("grid2_multi_2_op_output0", TensorProto.FLOAT, ["-1", 76, 76, 3, 80])
    grid2_multi_2_op = make_node( # 分数
        "Mul",
        inputs=["grid2_sigmoid_2_op_output0", "grid2_sigmoid_1_op_output0"],
        outputs=["grid2_multi_2_op_output0"])

    grid2_reshape_4_op_shape = make_tensor_value_info("grid2_reshape_4_op_shape", TensorProto.INT64, [1])
    grid2_reshape_4_op_shape_init = numpy_helper.from_array(np.array([-1, 76*76*3, 80], dtype=np.int64))
    grid2_reshape_4_op_shape_init.name = "grid2_reshape_4_op_shape"

    grid2_scores = make_tensor_value_info("grid2_scores", TensorProto.FLOAT, ["-1", 76*76*3, 80])
    grid2_reshape_4_op = make_node( # 分数reshape一下, grid2的最终分数
        "Reshape",
        inputs=["grid2_multi_2_op_output0", "grid2_reshape_4_op_shape"],
        outputs=["grid2_scores"])
    
    #### 拼装最终的positions和scores ####
    positions_concat_0_op_output = make_tensor_value_info("positions_concat_0_op_output", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 4])
    positions_concat_0_op = make_node(
        "Concat",
        inputs=["grid0_positions", "grid1_positions", "grid2_positions"],
        outputs=["positions_concat_0_op_output"],
        axis=-2)

    # positions_unsqueeze_op_output = make_tensor_value_info("positions_unsqueeze_op", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 4]) 
    positions_xywh = make_tensor_value_info("positions_xywh", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 4]) 
    positions_unsqueeze_op = make_node(
        "Unsqueeze",
        inputs=["positions_concat_0_op_output"],
        outputs=["positions_xywh"],
        axes=[-2],
    )

    # positions_concat_last_op = make_node(
    #     "Concat",
    #     inputs=["positions_unsqueeze_op_output"] * 80,
    #     outputs=["positions"],
    #     axis=-2,)

    positions_slice_0_op_start = make_tensor_value_info("positions_slice_0_op_start", TensorProto.INT64, [1])
    positions_slice_0_op_start_init = numpy_helper.from_array(np.array([0, 0, 0], dtype=np.int64))
    positions_slice_0_op_start_init.name = "positions_slice_0_op_start"

    positions_slice_0_op_end = make_tensor_value_info("positions_slice_0_op_end", TensorProto.INT64, [1])
    positions_slice_0_op_end_init = numpy_helper.from_array(np.array([(19*19+38*38+76*76)*3, 1, 1], dtype=np.int64))
    positions_slice_0_op_end_init.name = "positions_slice_0_op_end"

    positions_slice_0_op_axis = make_tensor_value_info("positions_slice_0_op_axis", TensorProto.INT64, [1])
    positions_slice_0_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64))
    positions_slice_0_op_axis_init.name = "positions_slice_0_op_axis" 
 
    positions_x1 = make_tensor_value_info("positions_x1", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 1])
    positions_slice_0_op = make_node(
        "Slice",
        inputs=["positions_xywh", "positions_slice_0_op_start", "positions_slice_0_op_end", "positions_slice_0_op_axis"],
        outputs=["positions_x1"]
    )

    positions_slice_1_op_start = make_tensor_value_info("positions_slice_1_op_start", TensorProto.INT64, [1])
    positions_slice_1_op_start_init = numpy_helper.from_array(np.array([0, 0, 1], dtype=np.int64))
    positions_slice_1_op_start_init.name = "positions_slice_1_op_start"

    positions_slice_1_op_end = make_tensor_value_info("positions_slice_1_op_end", TensorProto.INT64, [1])
    positions_slice_1_op_end_init = numpy_helper.from_array(np.array([(19*19+38*38+76*76)*3, 1, 2], dtype=np.int64))
    positions_slice_1_op_end_init.name = "positions_slice_1_op_end"

    positions_slice_1_op_axis = make_tensor_value_info("positions_slice_1_op_axis", TensorProto.INT64, [1])
    positions_slice_1_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64))
    positions_slice_1_op_axis_init.name = "positions_slice_1_op_axis"

    positions_y1 = make_tensor_value_info("positions_y1", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 1])
    positions_slice_1_op = make_node(
        "Slice",
        inputs=["positions_xywh", "positions_slice_1_op_start", "positions_slice_1_op_end", "positions_slice_1_op_axis"],
        outputs=["positions_y1"]
    )

    positions_slice_2_op_start = make_tensor_value_info("positions_slice_2_op_start", TensorProto.INT64, [1])
    positions_slice_2_op_start_init = numpy_helper.from_array(np.array([0, 0, 2], dtype=np.int64))
    positions_slice_2_op_start_init.name = "positions_slice_2_op_start"

    positions_slice_2_op_end = make_tensor_value_info("positions_slice_2_op_end", TensorProto.INT64, [1])
    positions_slice_2_op_end_init = numpy_helper.from_array(np.array([(19*19+38*38+76*76)*3, 1, 3], dtype=np.int64))
    positions_slice_2_op_end_init.name = "positions_slice_2_op_end"

    positions_slice_2_op_axis = make_tensor_value_info("positions_slice_2_op_axis", TensorProto.INT64, [1])
    positions_slice_2_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64))
    positions_slice_2_op_axis_init.name = "positions_slice_2_op_axis"

    positions_w = make_tensor_value_info("positions_w", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 1])
    positions_slice_2_op = make_node(
        "Slice",
        inputs=["positions_xywh", "positions_slice_2_op_start", "positions_slice_2_op_end", "positions_slice_2_op_axis"],
        outputs=["positions_w"]
    )

    positions_slice_3_op_start = make_tensor_value_info("positions_slice_3_op_start", TensorProto.INT64, [1])
    positions_slice_3_op_start_init = numpy_helper.from_array(np.array([0, 0, 3], dtype=np.int64))
    positions_slice_3_op_start_init.name = "positions_slice_3_op_start"

    positions_slice_3_op_end = make_tensor_value_info("positions_slice_3_op_end", TensorProto.INT64, [1])
    positions_slice_3_op_end_init = numpy_helper.from_array(np.array([(19*19+38*38+76*76)*3, 1, 4], dtype=np.int64))
    positions_slice_3_op_end_init.name = "positions_slice_3_op_end"

    positions_slice_3_op_axis = make_tensor_value_info("positions_slice_3_op_axis", TensorProto.INT64, [1])
    positions_slice_3_op_axis_init = numpy_helper.from_array(np.array([1, 2, 3], dtype=np.int64))
    positions_slice_3_op_axis_init.name = "positions_slice_3_op_axis"
    positions_h = make_tensor_value_info("positions_h", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 1])
    positions_slice_3_op = make_node(
        "Slice",
        inputs=["positions_xywh", "positions_slice_3_op_start", "positions_slice_3_op_end", "positions_slice_3_op_axis"],
        outputs=["positions_h"]
    )

    positions_x2 = make_tensor_value_info("positions_x2", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 1])
    positions_add_0_op = make_node(
        "Add",
        inputs=["positions_x1", "positions_w"],
        outputs=["positions_x2"]
    )

    positions_y2 = make_tensor_value_info("positions_y2", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 1])
    positions_add_1_op = make_node(
        "Add",
        inputs=["positions_y1", "positions_h"],
        outputs=["positions_y2"]
    )

    positions = make_tensor_value_info("positions", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 1, 4])
    positions_concat_1_op = make_node(
        "Concat",
        inputs=["positions_x1", "positions_y1", "positions_x2", "positions_y2"],
        outputs=["positions"],
        axis=-1
    )

    scores = make_tensor_value_info("scores", TensorProto.FLOAT, ["-1", (19*19+38*38+76*76)*3, 80])
    scores_concat_op = make_node(
        "Concat",
        inputs=["grid0_scores", "grid1_scores", "grid2_scores"],
        outputs=["scores"],
        axis=-2)

    
    #### Grid0添加操作节点 ####
    ngraph.node.append(grid0_transpose_0_op)
    ngraph.node.append(grid0_reshape_0_op)
    ngraph.node.append(grid0_slice_0_op)
    ngraph.node.append(grid0_sigmoid_0_op)
    ngraph.node.append(grid0_slice_1_op)
    ngraph.node.append(grid0_sigmoid_1_op)
    ngraph.node.append(grid0_slice_2_op)
    ngraph.node.append(grid0_sigmoid_2_op)
    ngraph.node.append(grid0_slice_3_op)
    ngraph.node.append(grid0_exponent_0_op)
    ngraph.node.append(grid0_multi_0_op)
    ngraph.node.append(grid0_tile_0_op)
    ngraph.node.append(grid0_reshape_1_op)
    ngraph.node.append(grid0_concat_0_op)
    ngraph.node.append(grid0_tile_1_op)
    ngraph.node.append(grid0_reshape_2_op)
    ngraph.node.append(grid0_concat_1_op)
    ngraph.node.append(grid0_concat_2_op)
    ngraph.node.append(grid0_add_0_op)
    ngraph.node.append(grid0_div_0_op)
    ngraph.node.append(grid0_div_1_op)
    ngraph.node.append(grid0_div_2_op)
    ngraph.node.append(grid0_sub_0_op)
    ngraph.node.append(grid0_concat_3_op)
    ngraph.node.append(grid0_reshape_3_op)
    ngraph.node.append(grid0_multi_1_op)
    ngraph.node.append(grid0_multi_2_op)
    ngraph.node.append(grid0_reshape_4_op)
    
    #### Grid1添加操作节点 ####
    ngraph.node.append(grid1_transpose_0_op)
    ngraph.node.append(grid1_reshape_0_op)
    ngraph.node.append(grid1_slice_0_op)
    ngraph.node.append(grid1_sigmoid_0_op)
    ngraph.node.append(grid1_slice_1_op)
    ngraph.node.append(grid1_sigmoid_1_op)
    ngraph.node.append(grid1_slice_2_op)
    ngraph.node.append(grid1_sigmoid_2_op)
    ngraph.node.append(grid1_slice_3_op)
    ngraph.node.append(grid1_exponent_0_op)
    ngraph.node.append(grid1_multi_0_op)
    ngraph.node.append(grid1_tile_0_op)
    ngraph.node.append(grid1_reshape_1_op)
    ngraph.node.append(grid1_concat_0_op)
    ngraph.node.append(grid1_tile_1_op)
    ngraph.node.append(grid1_reshape_2_op)
    ngraph.node.append(grid1_concat_1_op)
    ngraph.node.append(grid1_concat_2_op)
    ngraph.node.append(grid1_add_0_op)
    ngraph.node.append(grid1_div_0_op)
    ngraph.node.append(grid1_div_1_op)
    ngraph.node.append(grid1_div_2_op)
    ngraph.node.append(grid1_sub_0_op)
    ngraph.node.append(grid1_concat_3_op)
    ngraph.node.append(grid1_reshape_3_op)
    ngraph.node.append(grid1_multi_1_op)
    ngraph.node.append(grid1_multi_2_op)
    ngraph.node.append(grid1_reshape_4_op)

    #### Grid2添加操作节点 ####
    ngraph.node.append(grid2_transpose_0_op)
    ngraph.node.append(grid2_reshape_0_op)
    ngraph.node.append(grid2_slice_0_op)
    ngraph.node.append(grid2_sigmoid_0_op)
    ngraph.node.append(grid2_slice_1_op)
    ngraph.node.append(grid2_sigmoid_1_op)
    ngraph.node.append(grid2_slice_2_op)
    ngraph.node.append(grid2_sigmoid_2_op)
    ngraph.node.append(grid2_slice_3_op)
    ngraph.node.append(grid2_exponent_0_op)
    ngraph.node.append(grid2_multi_0_op)
    ngraph.node.append(grid2_tile_0_op)
    ngraph.node.append(grid2_reshape_1_op)
    ngraph.node.append(grid2_concat_0_op)
    ngraph.node.append(grid2_tile_1_op)
    ngraph.node.append(grid2_reshape_2_op)
    ngraph.node.append(grid2_concat_1_op)
    ngraph.node.append(grid2_concat_2_op)
    ngraph.node.append(grid2_add_0_op)
    ngraph.node.append(grid2_div_0_op)
    ngraph.node.append(grid2_div_1_op)
    ngraph.node.append(grid2_div_2_op)
    ngraph.node.append(grid2_sub_0_op)
    ngraph.node.append(grid2_concat_3_op)
    ngraph.node.append(grid2_reshape_3_op)
    ngraph.node.append(grid2_multi_1_op)
    ngraph.node.append(grid2_multi_2_op)
    ngraph.node.append(grid2_reshape_4_op)
    
    #### 组装三个Grid之后的节点 ####
    ngraph.node.append(positions_concat_0_op)
    ngraph.node.append(positions_unsqueeze_op)
    # ngraph.node.append(positions_concat_last_op)
    ngraph.node.append(positions_slice_0_op)
    ngraph.node.append(positions_slice_1_op)
    ngraph.node.append(positions_slice_2_op)
    ngraph.node.append(positions_slice_3_op)
    ngraph.node.append(positions_add_0_op)
    ngraph.node.append(positions_add_1_op)
    ngraph.node.append(positions_concat_1_op)
    ngraph.node.append(scores_concat_op)

    #### Grid0添加初始化节点 ####
    ngraph.initializer.extend([grid0_reshape_0_op_shape_init,
                               grid0_slice_0_op_start_init, grid0_slice_0_op_end_init, grid0_slice_0_op_axis_init,
                               grid0_slice_1_op_start_init, grid0_slice_1_op_end_init, grid0_slice_1_op_axis_init,
                               grid0_slice_2_op_start_init, grid0_slice_2_op_end_init, grid0_slice_2_op_axis_init,
                               grid0_slice_3_op_start_init, grid0_slice_3_op_end_init, grid0_slice_3_op_axis_init,
                               grid0_anchors_init,
                               grid0_tile_0_op_input0_init, grid0_tile_0_op_repeats_init, 
                               grid0_reshape_1_op_shape_init,
                               grid0_tile_1_op_input0_init, grid0_tile_1_op_repeats_init,
                               grid0_reshape_2_op_shape_init,
                               grid0_div_0_op_input1_init,
                               inputresolution_tuple_init,
                               two_init,
                               grid0_reshape_3_op_shape_init,
                               grid0_reshape_4_op_shape_init,])

    #### Grid1添加初始化节点 ####
    ngraph.initializer.extend([grid1_reshape_0_op_shape_init,
                               grid1_slice_0_op_start_init, grid1_slice_0_op_end_init, grid1_slice_0_op_axis_init,
                               grid1_slice_1_op_start_init, grid1_slice_1_op_end_init, grid1_slice_1_op_axis_init,
                               grid1_slice_2_op_start_init, grid1_slice_2_op_end_init, grid1_slice_2_op_axis_init,
                               grid1_slice_3_op_start_init, grid1_slice_3_op_end_init, grid1_slice_3_op_axis_init,
                               grid1_anchors_init,
                               grid1_tile_0_op_input0_init, grid1_tile_0_op_repeats_init,
                               grid1_reshape_1_op_shape_init,
                               grid1_tile_1_op_input0_init, grid1_tile_1_op_repeats_init,
                               grid1_reshape_2_op_shape_init,
                               grid1_div_0_op_input1_init,
                               grid1_reshape_3_op_shape_init,
                               grid1_reshape_4_op_shape_init,])

    #### Grid2添加初始化节点 ####
    ngraph.initializer.extend([grid2_reshape_0_op_shape_init,
                               grid2_slice_0_op_start_init, grid2_slice_0_op_end_init, grid2_slice_0_op_axis_init,
                               grid2_slice_1_op_start_init, grid2_slice_1_op_end_init, grid2_slice_1_op_axis_init,
                               grid2_slice_2_op_start_init, grid2_slice_2_op_end_init, grid2_slice_2_op_axis_init,
                               grid2_slice_3_op_start_init, grid2_slice_3_op_end_init, grid2_slice_3_op_axis_init,
                               grid2_anchors_init,
                               grid2_tile_0_op_input0_init, grid2_tile_0_op_repeats_init,
                               grid2_reshape_1_op_shape_init,
                               grid2_tile_1_op_input0_init, grid2_tile_1_op_repeats_init,
                               grid2_reshape_2_op_shape_init,
                               grid2_div_0_op_input1_init,
                               grid2_reshape_3_op_shape_init,
                               grid2_reshape_4_op_shape_init,])

    #### 组合三个Grid之后的初始化节点 ####
    ngraph.initializer.extend([positions_slice_0_op_start_init, positions_slice_0_op_end_init, positions_slice_0_op_axis_init,
                               positions_slice_1_op_start_init, positions_slice_1_op_end_init, positions_slice_1_op_axis_init,
                               positions_slice_2_op_start_init, positions_slice_2_op_end_init, positions_slice_2_op_axis_init,
                               positions_slice_3_op_start_init, positions_slice_3_op_end_init, positions_slice_3_op_axis_init,])
    
    #### 添加输入输出节点 ####
    ngraph.input.extend([origin_input_resolution])
    ngraph.output.extend([
        positions, # -1, 22743, 1, 4
        scores, # -1, 8, 22743, 80
        # positions_x1,
        # positions_y1,
        # positions_w,
        # positions_h
        ])

    return ngraph


if __name__ == "__main__":
    
    model = onnx.load("yolov3_dynamic_rename.onnx")
    print("+++++++++++++++++", model.opset_import)
    model_attrs = dict(
        ir_version=model.ir_version,
        opset_imports=model.opset_import,
        producer_version=model.producer_version,
        model_version=model.model_version
    )

    model = make_model(define_a_new_graph(model.graph), **model_attrs)
    # print(model.graph)
    checker.check_model(model)
    onnx.save(model, "yolov3_dynamic_postprocess.onnx")

