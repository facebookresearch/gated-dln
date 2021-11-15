def get_num_input_transformations_from_dataloader_name(name: str) -> int:
    return int(name.split("_transformations_")[0].split("_")[-1])


def get_num_classes_from_dataloader_name(name: str) -> int:
    return int(name.split("_classes_")[0].split("_")[-1])
