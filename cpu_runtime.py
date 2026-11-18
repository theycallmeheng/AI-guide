"""Explicit CPU thread tuning for the supported Ultralytics ONNX adapter."""
def configure_onnx_cpu(model, path, threads):
    import onnxruntime as ort
    backend = getattr(model.predictor.model, 'backend', model.predictor.model)
    session = getattr(backend, 'session', None)
    if session is None or session.get_providers() != ['CPUExecutionProvider']:
        return False
    if getattr(backend, 'use_io_binding', False):
        return False
    if not 1 <= threads <= 16:
        raise ValueError('ONNX CPU threads must be between 1 and 16')
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    backend.session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
    return True
