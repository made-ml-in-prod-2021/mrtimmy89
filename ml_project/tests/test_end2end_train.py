from src.data.make_dataset import read_data, dataset_split
from src.features.make_features import extract_target, full_transform
from src.models.fit_predict_model import train_model, predict_model, evaluate_model
from src.entities.split_parameters import SplittingParams


def test_train_pipeline(dataset_path: str) -> None:
    df = read_data(dataset_path)
    target = extract_target(df)
    splitting_params = SplittingParams()
    df_transformed = full_transform(df)
    X_train, X_test, y_train, y_test = dataset_split(
        df_transformed,
        target,
        splitting_params
    )
    model = train_model(X_train, y_train)
    pred_labels, pred_proba = predict_model(model, X_test)
    res = evaluate_model(y_test, pred_labels, pred_proba)
    assert len(res) == 5
    assert res["roc_auc_score"] > 0.5
