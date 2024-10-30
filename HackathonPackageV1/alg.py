from Roll_Inventory_Optimizer_Scoring import officialScorer
import pandas as pd

###Comparator Helper Method###
def compareOfficialScores(a,b):
    if a > b:
        return 1
    elif a < b:
        return -1
    else:
        return 0

###JSON###
def readJson():
    df = pd.read_json(t.staticPath)
    return df

def updateJson(df):
    df.to_json(t.bestSchedule, indent=4)

###Get new score###
def newScore(score):
    updateJson()
    (loss,breakdown) = officialScorer()
    return loss,breakdown

####Console print####
def console():
    pass
    # print(f"Epoch: {epoch} --- Start Val:{startVal} --- End Val:{jsonVal}")

    
# EX: Breakdown
# HoursBelowTMPOMinimumRuntime2hr                -0.000000
# totalOverlapHours                         -137211.111111
# gradeOOPViolationHoursTotal                -41000.000000
# gradeChangesConvertingHours                -40627.777778
# hoursBeyondEndBoundary                         -0.000000
# gradeMinViolationHoursTotal                -46000.000000
# proposedDemandViolationPenaltyTotal       -293473.056385
# HoursBelowTMPOMinimumRuntime8hr              -900.805556
# HoursBelowConvertingPOMinimumRuntime8hr     -2997.083333
# TM4GradeChangeOrderViolations                   0.000000
# gradeMaxViolationHoursTotal                 -7400.000000
# GradeChangeCountTM                            -30.000000
# GradeChangeCountConverting                    -55.000000
# rollsBelowMaxInventoryScoreTotal              -21.104258

###Main Algorithm###
def algorithm():
    jsonData = readJson()
    print(jsonData.shape)
    print(jsonData)
