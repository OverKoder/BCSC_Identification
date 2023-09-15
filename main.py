from src.data.load_data import load_data
from src.data.process_data import get_min_max_features, get_class_counts, get_all_from_classes
def main():

    print("Loading data...")
    data = load_data("../MasterThesisData/Scaled.h5ad")
    print("Done")
    min_f, max_f = get_min_max_features(data)
    print("Min features:", min_f)
    print("Max features:", max_f)

    class_counts = get_class_counts(data, 'main.ids')
    print("Main ids counts:", class_counts)

    class_counts = get_class_counts(data, 'main.ids.2')
    print("Main ids 2 counts:", class_counts)

    class_counts = get_class_counts(data, 'main.ids.3')
    print("Main ids 3 counts:", class_counts)

    new_data, new_labels = get_all_from_classes(data,2,[0,14,7,25],True)
    print("New data:", new_data.shape)
    print(new_labels)
if __name__ == "__main__":
    main()