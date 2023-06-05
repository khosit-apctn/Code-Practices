def count_pairs(number_list, power, distance):
    power_nums = [i**power for i in number_list]
    count = 0
    for i in range(len(number_list)):
        for j in range(i+1,len(number_list)):
            if number_list[i]%3==0 or number_list[j]%3==0 :
                if abs(power_nums[i]-power_nums[j]) <= 200:
                    count+=1
                    print( number_list[i] , number_list[j])
    
    print(count)

    print(power_nums)
    
# count_pairs([1,2,3,4,6,10], 3, 200)


def find_max_ngram (text, length):
    sub_strings = {}
    comp = ""
    # print("xbca">"bcax")
    for i in range(len(text)):
        if text[i:length+i] not in sub_strings:
            sub_strings[text[i:length+i]] = 0
        if text[i:length+i] in sub_strings:
            sub_strings[text[i:length+i]]=sub_strings[text[i:length+i]]+1
        # print(text[i:length+i])
    print(sub_strings)
    for val in sub_strings:
        if sub_strings[val] == max(sub_strings.values()):
            if comp == "": comp = val
            if comp > val: comp = val
    print(comp)

    # print(max(sub_strings.values()))

def find_max_ngram_correct (text, length):
    T_ngram = {}
    count = 0
    ans_ngram = ""
    # print("abcx"  < "bcax")
    for i in range(len(text)):

        ngram = text[i:i+length]
        # print(ngram)
        if ngram in T_ngram:
            T_ngram [ngram] += 1
        else:
            T_ngram[ngram]=1
        
        if T_ngram[ngram] > count:
            count = T_ngram[ngram]
            ans_ngram = ngram
        elif T_ngram[ngram] == count and ngram < ans_ngram:
            ans_ngram = ngram
    print(T_ngram)
    print(ans_ngram)

    
# find_max_ngram_correct("xbcxbcaxbcaxb", 4)


def check_perfect_list(numbers, distance):
    if len(numbers) < distance:
        return False
    for i in range(len(numbers)- distance +1):
        if abs(numbers[i]-numbers[i+1]) > distance or abs(numbers[i]-numbers[i+2])>distance:
            print(False)
    print(True)




def check_perfect_list_correct(numbers, distance):
    if len(numbers) < distance:
        return False

    for i in range(len(numbers) - distance +1):
        print(i)
        next = numbers[i+1:i+3]
        # print(next)
        for num in next:
            # print(num-numbers[i])
            if num - numbers[i] > distance:
                print(False)
                return False
    print(True)
    return True

# check_perfect_list_correct([1, 4, 3, 2, 6], 3)