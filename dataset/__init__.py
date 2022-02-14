from .hicodet import build_hico_det
__all__ = ['build_dataset']

DataFactory = dict(
    hico_det=build_hico_det
)


def build_dataset(dataset_name, root, anno_file):
    assert dataset_name in DataFactory.keys(),\
        f"{dataset_name} isn't in {DataFactory.keys()}"
    create_fn = DataFactory[dataset_name]
    return create_fn(root, anno_file)
