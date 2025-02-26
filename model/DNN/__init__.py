from .cnn import CNN


def make_model(config, n_feats:int):
    """
    Tạo mô hình dựa trên cấu hình được cung cấp.

    Args:
        config: Đối tượng chứa các tham số cấu hình mô hình.
        n_feats (int): Số lượng đặc trưng đầu vào (input features).

    Returns:
        model: Mô hình được tạo theo cấu hình.
    """

    if config.model == 'cnn':
        model = CNN.make(
            inoput_shape=n_feats,
            n_kernels=config.n_kernels,
            kernel_size = config.kernel_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            n_classes=len(config.class_labels),
            lr=config.lr
        ) 

    return model


_MODELS={
    'cnn': CNN
}

def load(config):
    """
    Tải mô hình đã được huấn luyện từ file checkpoint.

    Args:
        config: Đối tượng chứa các tham số cấu hình, bao gồm đường dẫn checkpoint.

    Returns:
        model: Mô hình đã tải từ file.
    """
    return _MODELS[config.model].load(
        path=config.checkpoint_path,
        name=config.checkpoint_name
    )