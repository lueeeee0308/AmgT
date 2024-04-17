with open('../hypre_test/data/preprocess_data.out', 'r') as f:
    file_lines = f.readlines()
    print("matrix,uscsr2bsr,usbsr2csr,cusparse_csr2bsr,cusparse_bsr2csr")
    words = [file_line.strip().split(' ') for file_line in file_lines]
    for word_ in words:
        if (word_[0] == "matrix"):
            step0=word_[len(word_)-1]
        if (word_[0] == "uscsr2bsr"):
            step1=word_[len(word_)-2]
        if (word_[0] == "usbsr2csr"):
            step2=word_[len(word_)-2]
        if (word_[0] == "cusparse"):
            if(word_[1] == "csr2bsr"):
                step3=word_[len(word_)-2]
        if (word_[0] == "cusparse"):
            if(word_[1] == "bsr2csr"):
                print(step0+','+step1+','+step2+','+step3+','+word_[len(word_)-2])
