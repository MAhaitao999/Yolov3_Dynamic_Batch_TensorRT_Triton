import onnx
import onnx_graphsurgeon as gs
import onnx
graph1 = gs.import_onnx(onnx.load("yolov3.onnx"))
tensors = graph1.tensors()
tensors["000_net"].name = "input"
tensors["082_convolutional"].name = "output0"
tensors["094_convolutional"].name = "output1"
tensors["106_convolutional"].name = "output2"
onnx.save(gs.export_onnx(graph1.cleanup().toposort()), "yolov3_dynamic_rename.onnx")
