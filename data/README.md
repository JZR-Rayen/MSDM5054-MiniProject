# Data

Raw datasets and complete processed training/test tables are not uploaded.

Download the official [Home Credit Default Risk competition data](https://www.kaggle.com/competitions/home-credit-default-risk/data), following Kaggle's access/rules. Place the files under a separate input root:

```text
input_root/
  home-credit-default-risk/
    application_train.csv
    application_test.csv
    bureau.csv
    bureau_balance.csv
    previous_application.csv
    installments_payments.csv
    POS_CASH_balance.csv
    credit_card_balance.csv
    sample_submission.csv
    HomeCredit_columns_description.csv
```

Pass `input_root` to `notebooks/reproduce.py data --data-root ...`. This route creates processed tables; you do not need to obtain them separately. See [notebooks/README.md](../notebooks/README.md#retrain-from-original-data) for the subsequent commands.

`notebooks/support/` contains derived results, saved OOF scores/labels/folds and compact plotting inputs. They enable default result verification but cannot replace training inputs. Required raw fingerprints and fixed definitions are checked before reconstruction. A contains application-table information and four shared derivatives; B adds 27 historical aggregates; C adds 39 behavioural features. Their shared widths are 126/153/192 including ID and label, not encoded model dimensions.
