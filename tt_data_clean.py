from db_config.tt_ywdata_config_tmp import *
from tqdm import tqdm


def read_txt():

    pretrain_txt = []
    with get_tt_ywdata_session() as s:
        res = s.query(tt_ywdata_content).filter(tt_ywdata_content.clean_txt != "").limit(30).all()
        for r in tqdm(res):
            clean_txt = r.clean_txt
            # print(clean_txt)
            pretrain_txt.append(clean_txt)

    generate_pickle("./data/pretrain_txt1229.pkl", pretrain_txt)

def write_txt(file_name, text_list):
    # text_list = map(lambda x: x+"\n", text_list)
    # print(list(text_list))

    train_text = []
    validation_text = []
    for idx, text_line in enumerate(text_list):
        if idx <= 20:
            train_text.append(text_line)
        else:
            validation_text.append(text_line)

    with open('./data/tt_text/train_'+file_name, 'a+') as f:
        for text_line in train_text:
            f.write(text_line+"\n")

    with open('./data/tt_text/validation_'+file_name, 'a+') as f:
        for text_line in validation_text:
            f.write(text_line+"\n")

if __name__ == '__main__':
    read_txt()

    # tmp_text_list = load_pickle("./data/pretrain_txt1229.pkl")
    tmp_text_list = load_pickle("./data/pretrain_txt1229.pkl")


    write_txt("pretrain_txt1229.txt", tmp_text_list)