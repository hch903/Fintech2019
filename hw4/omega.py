import numpy 
import numpy.random as nrand
import pandas as pd
FREE_RATE = 0.025
def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return numpy.sum(diff ** order) / len(returns)

def preprocess(list,period):
    oldlist = numpy.array(list)
    newlist = []
    for i in range(0,len(oldlist),period):
        if(i+period<len(oldlist)):
            e = numpy.mean(oldlist[i:i+period])
        else:
            e = numpy.mean(oldlist[i:])
        newlist.append(e)
    newlist = numpy.array(newlist)
    newlist = norm(newlist)
    return newlist
def norm(raw_list):
    e = numpy.mean(raw_list)
    std_list = (raw_list-e)/e
    return std_list

def omega_ratio(returns, rf, target=0):
    er = numpy.mean(returns)
    return (er - rf) / lpm(returns, target, 1)
def main():
    df = pd.read_excel("output.xlsx")
    week_dict = {}
    month_dict = {}
    # a = numpy.array(df.iloc(1)[2])[1:]
    # print(numpy.array(df.iloc(1)[2])[1:][::-1])
    # exit()
    for i in range(2,46):
        mylist = numpy.array(df.iloc(1)[i])
        symbol = mylist[0]
        rev_list = mylist[1:][::-1]
        week_dict[symbol] = preprocess(rev_list,5) # 5 day per week
        month_dict[symbol] = preprocess(rev_list,20)# 4 week per month
    
    for symbol,val in week_dict.items():
        week_dict[symbol] = omega_ratio(val,FREE_RATE)
    for symbol,val in month_dict.items():
        month_dict[symbol] = omega_ratio(val,FREE_RATE)
    week_rank = sorted(week_dict,key = week_dict.get,reverse = True)
    month_rank = sorted(month_dict,key = month_dict.get,reverse = True)
    print("top20 sort by week",week_rank[:20])
    print("top20 sort by month",month_rank[:20])
    week_rank = [[i+1,x] for i,x in enumerate(week_rank)]
    month_rank = [ [i+1,x] for i,x in enumerate(month_rank)]
    w_df_obj = pd.DataFrame(week_rank,columns = ['Ranking','ETF_Symbol'])
    m_df_obj = pd.DataFrame(month_rank,columns = ['Ranking','ETF_Symbol'])
    w_df_obj.to_excel("week_omega_rank.xlsx")
    m_df_obj.to_excel("month_omege_rank.xlsx")

    
if __name__ == '__main__':
    main()