local dr = "/data/";
local sav_dir = "/data/preprocessed/";
local emb_dir = "~/.embeddings/";

{
    data: {
        # qa
        train_annotations: dr + "v2_mscoco_train2014_annotations.json",
        val_annotations: dr + "v2_mscoco_val2014_annotations.json",
        train_questions: dr + "v2_OpenEnded_mscoco_train2014_questions.json",
        val_questions: dr + "v2_OpenEnded_mscoco_val2014_questions.json",
        train_qa_result_file: sav_dir + "qa_train.pkl",
        val_qa_result_file: sav_dir + "qa_val.pkl",
        answer_vocab_result_file: sav_dir + "answer_vocab.json",
        max_answers: 1000,
        # image
        train_images: dr + "train2014",
        val_images: dr + "val2014",
        train_images_result_file: sav_dir + "train_images.hdf5",
        train_filenames_result_file: sav_dir + "train_filenames.json",
        val_images_result_file: sav_dir + "val_images.hdf5",
        val_filenames_result_file: sav_dir + "val_filenames.json",
        # vocab
        vocab_result_file: sav_dir + "vocab.txt",
        # embeddings
        use_pretrained_embeddings: true,
        pretrained_embeddings: emb_dir + "GoogleNews-vectors-negative300.bin",
        embeddings_result_file: sav_dir + "saved_embeddings.pkl"
    },
    model: {
        emb_size: 300,
        hidden_size: 1024,
        dropout_rate: 0.1,
        image_emb_size: 4096,
        n_classes: $.data.max_answers
    },
    training : {
        n_epochs: 5,
        lr: 5e-4,
        train_batch_size: 128,
        val_batch_size: 128
    },
}