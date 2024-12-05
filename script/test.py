from transformers import AutoTokenizer, AutoModel

def main():
    # モデル名を指定
    model_name = "google/gemma-2-9b"

    # ローカルに保存するパス
    local_dir = "../model/gemma-2-9b"

    # モデルとトークナイザーをダウンロードして保存
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token="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")
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    tokenizer.save_pretrained(local_dir)
    model.save_pretrained(local_dir)

    print("Done.")
if __name__ == '__main__':
    main()


