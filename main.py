import DataPreProgress as dpp
import pandas as pd
import Analysis as ana
from Analysis import bank_processed


def load_and_process_data():
    # 导入数据
    bank = pd.read_csv('bank/bank.csv', sep=';')
    bankFull = pd.read_csv('bank/bank-full.csv', sep=';')
    bankAdditional = pd.read_csv('bank-additional/bank-additional.csv', sep=';')
    bankAdditionalFull = pd.read_csv('bank-additional/bank-additional-full.csv', sep=';')

    # 数据处理
    bank = dpp.data_processing(bank)
    dpp.write_to_csv(bank, 'bank/bank_processed.csv')

    bankFull = dpp.data_processing(bankFull)
    dpp.write_to_csv(bankFull, 'bank/bank-full_processed.csv')

    bankAdditional = dpp.data_processing(bankAdditional)
    dpp.write_to_csv(bankAdditional, 'bank-additional/bank-additional_processed.csv')

    bankAdditionalFull = dpp.data_processing(bankAdditionalFull)
    dpp.write_to_csv(bankAdditionalFull, 'bank-additional/bank-additional-full_processed.csv')

    return bank, bankFull, bankAdditional, bankAdditionalFull

def main():
    load_and_process_data()
    ana.analysis(bank_processed)
if __name__ == "__main__":
    main()