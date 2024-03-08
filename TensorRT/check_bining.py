# import tensorrt as trt

# # Đọc tệp engine TensorRT
# engine_file_path = 'TensorRT/model_rt/model_cls_bottle.trt'
# with open(engine_file_path, "rb") as f:
#     engine_data = f.read()


# # Deserialize engine
# runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
# engine = runtime.deserialize_cuda_engine(engine_data)

# # In ra các tên binding
# print("Binding names:")
# for i in range(engine.num_bindings):
#     print(engine.get_binding_name(i))


# print("Output binding names:")
# for i in range(engine.num_bindings):
#     binding_name = engine.get_binding_name(i)
#     # Kiểm tra xem binding có phải là một output hay không
#     if engine.binding_is_input(binding_name):
#         continue  # Bỏ qua nếu là input
#     print(binding_name)


import tensorrt as trt

# Load your TensorRT engine
with open("TensorRT/model_rt/model_cls_label.trt", "rb") as f:

    engine_bytes = f.read()
runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_bytes)
# Get binding names
input_binding_names = []
output_binding_names = []
for i in range(engine.num_bindings):
    if engine.binding_is_input(i):
        input_binding_names.append(engine.get_binding_name(i))
    else:
        output_binding_names.append(engine.get_binding_name(i))

# Get binding shapes
for i, name in enumerate(input_binding_names):
    binding_index = engine.get_binding_index(name)
    shape = engine.get_binding_shape(binding_index)
    print("Input {}: {} - Shape: {}".format(i, name, shape))
