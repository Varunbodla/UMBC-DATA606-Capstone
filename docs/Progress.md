## Cross-Linguistic Communication - A Natural Language Processing Approach
## Project Progress

Steps - 
1. Download the data from the source https://www.cfilt.iitb.ac.in/iitb_parallel/ and upload to the google drive.
2. Open colab and create a new notebook.
3. Mount the google drive to the notebook.
4. Importing the required libraries.

   ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/45b90070-78fe-4d81-9a1a-ffbaf232328c)

5. Extracting all the files from the zip file using zipfile library.

   ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/f986d715-dc33-4bc8-96f7-b25949c51ee5)

6. Read the data from the file and store the english text data in the list.

   ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/212796e6-4a71-421f-a126-36b1b556661e)

7. Read the data from the file and store the hindi text data in the list.

   ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/77c2a9c0-fee3-484c-b5a3-0d96b5c7d786)

9. Checking for the length of the data in both the languages.

   ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/5d7f84db-9a6e-4af9-bd5e-913af009ae88)

10. Creating a dataframe for the above english and hindi text data.

    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/566885f1-5d80-4d0e-ac22-31b1c7178378)

11. Processing and cleaning the text data using the below functions.

    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/0149f9e5-6f1c-44b2-83e1-8e7531fe3754)


    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/8631525c-dacf-48eb-bda8-4fbfd66d86f2)

13. Applying the function to both the language data for the created dataframe.

    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/a47e3a84-e9cc-48c7-b150-ef9283f3eec7)

14. Calculating the number of words in each entry of the 'hindi' and 'english' columns. These lengths are extracted to check the distribution of word lengths.

    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/a804e286-ed0e-4232-b45f-7dad02d10adc)

15. Plotting the distribution of number of words for each entry for both the languages.

    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/3162837a-2eec-46bd-8399-5de9de14640b)

    It can be observed from the above histograms that the distribution of word lengths for both hindi and english are skewed towards right. The following things can be inferred.

    There are more sentences with less number of words in text in both english and hindi languages for the given data.

16. Calculating the percentiles for english and hindi word lengths.

    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/5a5f82ce-f6c5-447a-9504-9cee08f6494a)
    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/73250233-cb0d-4eaf-9835-b944ad5fdfd0)

17. Preparing the data for modelling by appending <start> and <end> tokens.

    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/4d5448a6-c25b-4d14-8334-41f759b563c3)

18. Splitting the data into train and test in the ratio of 75.:25 respectively.

    ![image](https://github.com/Varunbodla/UMBC-DATA606-Capstone/assets/85016388/d71b658b-3992-4735-9755-fc393e29fb17)


