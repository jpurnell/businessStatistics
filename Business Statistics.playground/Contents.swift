import UIKit
import Foundation

protocol Numeric: Comparable {}

extension Float: Numeric {}
extension Double: Numeric {}
extension Int: Numeric {}

// Ranking elements in an array – not critical for this project
extension Array where Element: Numeric {
    
    internal func rank() -> [Double] {
        let sorted = self.sorted(by: {$0 > $1})
        var rankArray: [Double] = []
        for i in 0..<self.count {
            rankArray.append(Double(sorted.firstIndex(of: self[i])! + 1))
        }
        var counts:[Double: Int] = [:]
        rankArray.forEach({counts[$0, default: 0] += 1})
        var tieAdjustment: Double = 0.0
        for count in counts {
            tieAdjustment += (((count.key * count.key * count.key) - count.key) / 12)
        }
        for (index, absoluteRank) in rankArray.enumerated() {
            let n = Double(counts[absoluteRank]!)
            rankArray[index] = (((n * absoluteRank) + (((n - 1) * n) / 2)) / n)
        }
        return rankArray
    }
    
    internal func reverseRank() -> [Double] {
        let sorted = self.sorted(by: {$0 < $1})
        var rankArray: [Double] = []
        for i in 0..<self.count {
            rankArray.append(Double(sorted.firstIndex(of: self[i])! + 1))
        }
        var counts:[Double: Int] = [:]
        rankArray.forEach({counts[$0, default: 0] += 1})
        var tieAdjustment: Double = 0.0
        for count in counts {
            tieAdjustment += (((count.key * count.key * count.key) - count.key) / 12)
        }
        for (index, absoluteRank) in rankArray.enumerated() {
            let n = Double(counts[absoluteRank]!)
            rankArray[index] = (((n * absoluteRank) + (((n - 1) * n) / 2)) / n)
        }
        return rankArray
    }
    
    internal func tauAdjustment() -> Double {
        let sorted = self.sorted(by: {$0 > $1})
        var rankArray: [Double] = []
        for i in 0..<self.count {
            rankArray.append(Double(sorted.firstIndex(of: self[i])! + 1))
        }
        var counts:[Double: Int] = [:]
        rankArray.forEach({counts[$0, default: 0] += 1})
        var tieAdjustment: Double = 0.0
        for count in counts {
            var adjustment: Double = 0
            if count.value > 1 {
                adjustment = ((Double((count.value * count.value * count.value) - count.value)) / 12.0)
            }
            tieAdjustment += adjustment
        }
        return tieAdjustment
    }
}

// MARK: - Descriptive Statistics
// We use descriptive statistics to describe the observations that we've seen over time. This helps us to understand, statistically, what we should expect of a standard, repeating process, helping us to know what the range of outcomes should be. By giving us the average and the standard deviation, we can look at any individual outcome and know if it is expected, or if it's outside of the range of what we would expect, either good or bad, and give us an insight on when we should ask more questions about why something may have happened.

// MARK – Basics – Mean and Standard Deviation calculations
// We default to using Doubles
// "Mean" and "Average" tend to mean the same thing. For all observations, take the sum and divide by the number of observations
public func mean(_ x: [Double]) -> Double {
    guard x.count > 0 else {
        return 0
    }
    return (x.reduce(0.0, +) / Double(x.count))
}

public func average(_ x: [Double]) -> Double {
    return mean(x)
}

// Median calculates, for given sample, what number sits in between the upper 50% and the lower 50% of samples.
public func median(_ x: [Double]) -> Double {
    if x.count == 0 { return 0 } else {
        if x.count % 2 == 0 {
            let l = x.count / 2
            let u = l + 1
            let lower = x[l]
            let upper = x[u]
            let num = lower + upper
            return num  / 2
            
        } else {
            let median = ((x.count + 1) / 2)
            return x[median]
        }
    }
}

// Mode is the number that appears most frequently in a given set of samples
public func mode<T>(_ x: [T]) -> T {
    let counted = NSCountedSet(array: x)
    let max = counted.max { counted.count(for: $0) < counted.count(for: $1)}
    return max as! T
}



// MARK: - Variance summarizes the how wide the differences are between observations and the mean. This is just a step towards the standard deviation, which is more useful mathematically. Once we have the mean (average), we square the difference of each observation that we used to calculate the mean from the mean. We then add those all up to get the Sum of Squared Average Difference. We use the square here so that the negative differences don't offset the positive differences and just give us 0. This Sum of Squared Average Difference is then averaged itself to give us the Variance.
public func sumOfSquaredAvgDiff(_ values: [Double]) -> Double {
    return values.map{ pow($0 - mean(values), 2.0)}.reduce(0, {$0 + $1})
}

// Default to using sample, which is more likely, but enable population when available
public enum Population: String {
    case population
    case sample
}

// MARK: - The variance, when we have the entire population of values and not a sample, is the Sum of Squared Average Difference, averaged over the total number of observations
public func varianceP(_ values: [Double]) -> Double {
    return sumOfSquaredAvgDiff(values)/Double(values.count)
}
 
// When we are working with a subset (sample) of the total number of observations, we use the sum of squared average differences, but divide it by one fewer than the number of observations. If there are fewer than 30 observations in the sample, we use the T-Distribution of the Variance (varianceTDist)
public func varianceS(_ values: [Double]) -> Double {
    if values.count < 30 {
        return varianceTDist(values)
    }
    let degreesOfFreedom = values.count - 1
    return sumOfSquaredAvgDiff(values)/Double(degreesOfFreedom)
}

public func variance(_ values: [Double], _ pop: Population = .sample) -> Double {
    switch pop {
        case .population:
            return varianceP(values)
        default:
            return varianceS(values)
    }
}

public func varianceTDist(_ values: [Double]) -> Double {
    if values.count > 30 { return variance(values) }
    return Double((values.count - 1) / (values.count - 3))
}

public func tStatistic(x: Double, mean: Double = 0.0, stdErr: Double = 1.0) -> Double {
    return ((x - mean) / stdErr)
}

//MARK: - Standard Deviation helps us understand the dispersion of our observations. This is critical, because if we have a mean observation of 100, that could be because we have 1000 observations that are between 95 and 105, or it could be because we have 1000 observations between 80 and 120. This allows us to know when a single observation is outside of what we should expect to be the "natural" range, given the data we've already collected. If we saw a "110" come through in the first scenario (a typical range between 95 - 105), we might have some questions, but if it were in the second scenario (80-120), we probably wouldn't care.

// Standard deviation for a population, used when you have all observations of a set, is just the square root of the population variance calculated above.
public func stdDevP(_ values: [Double]) -> Double {
    return sqrt(varianceP(values))
}

// Standard deviation for a sample, used when you do not have all observations of the population, e.g. last 30 day calculations is the square root of the *sample* variance calculated above.
public func stdDevS(_ values: [Double]) -> Double {
    return sqrt(varianceS(values))
}


public func stdDev(_ values: [Double], _ pop: Population = .sample) -> Double {
    switch pop {
        case .population:
            return stdDevP(values)
        default:
            return stdDevS(values)
    }
}

public func stdDevTDist(_ values: [Double]) -> Double {
    return sqrt(varianceTDist(values))
}


// MARK: - Advanced Descriptors
// Advanced Descriptors give us a better sense of the "shape" of the overall data, helping us understand if outliers are making our basic descriptors not tell the whole story. Skew helps us identify cases where maybe most results are on one side of the average, but a really big outlier on the other side of the average is changing the numbers (e.g. 999 observations of 1, but one observation of 100,000 makes your average 1,001)
public func coefficientOfSkew(mean: Double, median: Double, stdDev: Double) -> Double {
    return (3 * (mean - median))/stdDev
}

public func coefficientOfSkew(_ values: [Double]) -> Double {
    return coefficientOfSkew(mean: mean(values), median: median(values), stdDev: stdDev(values))
}

public func coefficientOfVariation(_ stdDev: Double, mean: Double) -> Double {
    return (stdDev / mean) * 100
}

public func descriptives(_ values: [Double]) -> (mean: Double, stdDev: Double, skew: Double, cVar: Double) {
    let mu = mean(values)
    let stDev = stdDev(values)
    let skew = coefficientOfSkew(values)
    let coVar = coefficientOfVariation(stDev, mean: mu)
    return (mu, stDev, skew, coVar)
}

public func PercentileLocation<T: Comparable>(_ percentile: Int, values: [T]) -> T {
    return values.sorted()[(values.count + 1)*(percentile / 100)]
}

// Spearman's Rho is not critcal
public func spearmansRho(_ independent: [Double], vs variable: [Double]) -> Double {
    var sigmaD: Double = 0.0
    let sigmaX = (pow(Double(independent.count), 3.0) - Double(independent.count)) / 12 - independent.tauAdjustment()
    let sigmaY = (pow(Double(variable.count), 3.0) - Double(variable.count)) / 12 - variable.tauAdjustment()
    
    let independentRank = independent.rank()
    let variableRank = variable.rank()
    
    for i in 0..<independent.count {
        sigmaD += Double((independentRank[i] - variableRank[i]) * (independentRank[i] - variableRank[i]))
    }
    let rho = (sigmaX + sigmaY - sigmaD) / (2.0 * sqrt((sigmaX * sigmaY)))
    return rho
}

public func zStatistic(x: Double, mean: Double = 0.0, stdDev: Double = 1.0) -> Double {
    return ((x - mean) / stdDev)
}

public func percentile(zScore z: Double) -> Double {
    return 0.5 * (1 + erf(z / sqrt(2.0)))
}

// erfInv allows us to calculate the zScore for a desired Area under a normal curve without having to rely on a lookup table
// https://stackoverflow.com/questions/36784763/is-there-an-inverse-error-public function-available-in-swifts-foundation-import
public func erfinv(y: Double) -> Double {
    let center = 0.7
    let a = [ 0.886226899, -1.645349621,  0.914624893, -0.140543331]
    let b = [-2.118377725,  1.442710462, -0.329097515,  0.012229801]
    let c = [-1.970840454, -1.624906493,  3.429567803,  1.641345311]
    let d = [ 3.543889200,  1.637067800]
    if abs(y) <= center {
        let z = pow(y,2)
        let num = (((a[3]*z + a[2])*z + a[1])*z) + a[0]
        let den = ((((b[3]*z + b[2])*z + b[1])*z + b[0])*z + 1.0)
        var x = y*num/den
        x = x - (erf(x) - y)/(2.0/sqrt(.pi)*exp(-x*x))
        x = x - (erf(x) - y)/(2.0/sqrt(.pi)*exp(-x*x))
        return x
    }
    else if abs(y) > center && abs(y) < 1.0 {
        let z = pow(-log((1.0-abs(y))/2),0.5)
        let num = ((c[3]*z + c[2])*z + c[1])*z + c[0]
        let den = (d[1]*z + d[0])*z + 1
        // should use the sign public function instead of pow(pow(y,2),0.5)
        var x = y/pow(pow(y,2),0.5)*num/den
        x = x - (erf(x) - y)/(2.0/sqrt(.pi)*exp(-x*x))
        x = x - (erf(x) - y)/(2.0/sqrt(.pi)*exp(-x*x))
        return x
    } else if abs(y) == 1 {
        return y * Double(Int.max)
    } else {
        return .nan
    }
}

public func zScore(percentile: Double) -> Double {
    return sqrt(2.0) * erfinv(y: ((2.0 * percentile) - 1))
}

public func zScore(ci: Double) -> Double {
    let lowProb = (1 - ci) / 2
    let highProb = 1 - lowProb
    return zScore(percentile: highProb)
}

public func percentile(x: Double, mean: Double, stdDev: Double) -> Double {
    return percentile(zScore: zStatistic(x: x, mean: mean, stdDev: stdDev))
}

// Probability Distribution public functon
public func normalPDF(x: Double, mean: Double = 0.0, stdDev: Double = 1.0) -> Double {
    let sqrt2Pi = sqrt(2 * Double.pi)
    let xMinusMeanSquared = (x - mean) * (x - mean)
    let stdDevSquaredTimesTwo = 2 * stdDev * stdDev
    let numerator = exp(-xMinusMeanSquared / stdDevSquaredTimesTwo)
    return numerator / (sqrt2Pi * stdDev)
}

// Normal Cumulative Distribution public function
public func normalCDF(x: Double, mean: Double = 0.0, stdDev: Double = 1.0) -> Double {
    return (1 + erf((x - mean) / sqrt(2.0) / stdDev)) / 2
}

public func inverseNormalCDF(p: Double, mean: Double = 0.0, stdDev: Double = 1.0, tolerance: Double = 0.00001) -> Double {
    if mean != 0.0 || stdDev != 1.0 {
        return mean + stdDev * inverseNormalCDF(p: p)
    }
    
    var lowZ = -10.0
    var midZ = 0.0
    var midP = 0.0
    var hiZ = 10.0
    
    while hiZ - lowZ > tolerance {
        midZ = (lowZ + hiZ) / 2
        midP = normalCDF(x: midZ)
        
        if midP < p {
            lowZ = midZ
        }
        else if midP > p {
            hiZ = midZ
        }
        else {
            break
        }
    }
    return midZ
}

public func uniformCDF(x: Double) -> Double {
    if x < 0 {
        return 0.0
    }
    else if x < 1 {
        return x
    }
    
    return 1.0
}

public func bernoulliTrial(p: Double) -> Int {
    if drand48() < p {
        return 1
    }
    return 0
}

public func binomial(n: Int, p: Double) -> Int {
    var sum = 0
    for _ in 0..<n {
        sum += bernoulliTrial(p: p)
    }
    return sum
}

public func confidenceInterval(mean: Double, stdDev: Double, z: Double, popSize: Int) -> (low: Double, high: Double) {
    return (low: mean - (z * stdDev/sqrt(Double(popSize))), high: mean +  (z * stdDev/sqrt(Double(popSize))))
}

public func confidenceInterval(ci: Double, values: [Double]) -> (low: Double, high: Double) {
    // Range in which we can expect the population mean to be found with x% confidence
    let lowProb = (1 - ci) / 2
    let highProb = 1 - lowProb
    
    let lowValue = inverseNormalCDF(p: lowProb, mean: mean(values), stdDev: stdDev(values))
    let highValue = inverseNormalCDF(p: highProb, mean: mean(values), stdDev: stdDev(values))
    
    return (lowValue, highValue)
}

// What we care about is when an observation is above or below our particular confidence interval for a given range
public func interestingObservation(observation x: Double, values: [Double], confidenceInterval ci: Double) -> Bool {
    let ciRange = confidenceInterval(ci: ci, values: values)
    if x <= ciRange.low || x >= ciRange.high {
        return true
    }
    return false
}

public func covariancePopulation(x: [Double], y: [Double]) -> Double {
    let xCount = Double(x.count)
    let yCount = Double(y.count)
    
    let xMean = average(x)
    let yMean = average(y)
  
    if xCount == 0 { return 0 }
    if xCount != yCount { return 0 }
    
        var sum: Double = 0
        
        for (index, xElement) in x.enumerated() {
            let yElement = y[index]
            sum += ((xElement - xMean) * (yElement - yMean))
        }
        return sum / xCount
}

public func factorial(_ N: Int) -> Int {
    if N < 1 { return 1 }
    var result = 1
    for i in 1...N {
        result = i * result
    }
    return result
}

public func chi2cdf(x: Double, dF: Double, slices: Double = 1000) -> Double {
    var returnValue: Double = 0
    let moment = 1 / slices
    for x in stride(from: moment, to: x, by: moment) {
        let dF: Double = dF
        let topLeft = pow(x, ((dF-2)/2))
        let topRight = 1 / exp(x/2)
        let bottomLeft = pow(2.0, dF/2)
        let bottomRight = tgamma(dF/2)
        
        let top = topLeft * topRight
        let bottom = bottomLeft * bottomRight
        returnValue += top / bottom
    }
    return returnValue / slices
}

public func sampleCorrelationCoefficient(_ independent: [Double], vs variable: [Double]) -> Double {
    let numerator = covariancePopulation(x: independent, y: variable)
    let denominator = (stdDev(independent) * stdDev(variable))
    let r = numerator / denominator
    return r
}

public func fisher(_ r: Double) -> Double {
    return (log((1 + r) / (1 - r)) / 2)
}

public func tStatistic(_ rho: Double, dFr: Double) -> Double {
    let tStatistic = rho * sqrt(dFr / (1 - (rho *  rho)))
    return tStatistic
}

public func tStatistic(_ independent: [Double], _ variable: [Double]) -> Double {
    return tStatistic(spearmansRho(independent, vs: variable), dFr: Double(independent.count - 2))
}

public func pValueStudent(_ tValue: Double, dFr: Double) -> Double {
    let rhoTop = tgamma((dFr + 1) / 2.0)
    let rhoBot = sqrt(dFr * Double.pi) * tgamma(dFr / 2)
    let left = rhoTop / rhoBot
    let center = (1 + ((tValue * tValue)/dFr))
    let centEx = -1 * ((dFr + 1) / 2)
    let right = pow(center, centEx)
    let pValueStudent = left * right
    return pValueStudent
}

public func pValue(_ independent: [Double], _ variable: [Double]) -> Double {
    return pValueStudent(tStatistic(independent, variable), dFr: Double(independent.count - 2))
}

public func derivativeOf(_ fn: (Double) -> Double, at x: Double) -> Double {
    let h: Double = 0.0000001
    return (fn(x + h) - fn(x) / h)
}

public func zScore(rho: Double, items: Int) -> Double {
    return sqrt(Double(items - 3)/1.06) * fisher(rho)
}

public func zScore(fisherR: Double, items: Int) -> Double {
    return sqrt(Double(items - 3)/1.06) * fisherR
}

public func rho(from fisherR: Double) -> Double {
    let top = exp(2.0 * fisherR) - 1
    let bottom = exp(2.0 * fisherR) + 1
    return top / bottom
}

public func correlationBreakpoint(_ items: Int, probability: Double) -> Double {
    let zComponents = sqrt(Double(items - 3)/1.06)
    let fisherR = inverseNormalCDF(p: probability) / zComponents
    return rho(from: fisherR)
}

extension Int {
    public func factorial() -> Int {
        if self >= 0 {
            return self == 0 ? 1 : self * (self - 1).factorial()
        } else {
            return 0
        }
    }
}

public func permutations(_ n: Int, r: Int) -> Int {
    let perms = n - r
    return n.factorial() / perms.factorial()
}

public func combinations(_ n: Int, r: Int) -> Int {
    let perms = n - r
    return (n.factorial() / (r.factorial() * perms.factorial()))
}

public func binomial(n: Int, x: Int, prob: Double) -> Double {
    return Double(combinations(n, r: x)) * pow(prob, Double(x)) * pow((1 - prob), Double(n - x))
}

public func meanBinomial(n: Int, prob: Double) -> Double {
    return Double(n) * prob
}

public func varianceBinomial(n: Int, prob: Double) -> Double {
    return Double(n) * prob * (1 - prob)
}

public func stdDevBinomial(n: Int, prob: Double) -> Double {
    return sqrt(varianceBinomial(n: n, prob: prob))
}

// MARK: - Hypergeometric Distribution: If a sample is selected without replacement from a known finite population and contains a relatively large proportion of the population, such that the probability of a success is measurably altered from one selection to the next, the hypergeometric distribution should be used.
// Assume a stable has total = 10 horses, and r = 4 of them have a contagious disesase, what is the probability of selecting a sample of n = 3 in which there are x = 2 diseased horses?
public func hypergeometric(total: Int, r: Int, n: Int, x: Int) -> Double {
    return Double(combinations(r, r: x) * combinations(total - r, r: n - x)) / Double(combinations(total, r: n))
}

// MARK: - Poisson Distribution: Measures the probability of a random event happening over some interval of some time or space. Assumes two things: 1) Probability of the occurence is constant for any two intervals of time or space. 2) The occurence of the even in any interval is independent of the occurence in any other interval
public func poisson(_ x: Int, µ: Double) -> Double {
    let numerator = pow(µ, Double(x)) * exp(-1 * µ)
    let denominator = x.factorial()
    return numerator / Double(denominator)
}

public func standardError(_ stdDev: Double, observations n: Int) -> Double {
    return stdDev / sqrt(Double(n))
}

public func standardError(_ x: [Double]) -> Double {
    return standardError(stdDev(x, .sample), observations: x.count)
}

// Corrected with the finite population correction factor. To be used if the sample size is more than 5% of the population
public func correctedStdErr(_ x: [Double], population: Int) -> Double {
    let percentage = Double(x.count / population)
    if percentage >= 0.05 { return standardError(x) } else {
        let num = population - x.count
        let den = population - 1
        return standardError(x) * (sqrt(Double(num/den)))
    }
}

public func estMean(probabilities x: [Double]) -> Double {
    return x.reduce(0.0, +) / Double(x.count)
}

public func standardErrorProbabilistic(_ prob: Double, observations n: Int) -> Double {
    if prob > 1 { return 0 } else {
        return sqrt(prob * (1 - prob) / Double(n))
    }
}

public func standardErrorProbabilistic(_ prob: Double, observation n: Int, totalObservations total: Int) -> Double {
    if Double(n/total) <= 0.05 {
        return standardErrorProbabilistic(prob, observations: n)
    } else {
        return standardErrorProbabilistic(prob, observations: n) * (sqrt(Double ((total - n)/(total - 1))))
    }
}

public func confidenceIntervalProbabilistic(_ prob: Double, observations n: Int, ci: Double) -> (low: Double, high: Double) {
    let lowProb = (1 - ci) / 2
    let highProb = 1 - lowProb
    let standardError = standardErrorProbabilistic(prob, observations: n)
    let z = zScore(percentile: highProb)
    let lowerCI = prob - (z * standardError)
    let upperCI = prob + (z * standardError)
    return (low: lowerCI, high: upperCI)
}

public func requiredSampleSize(z: Double, stdDev: Double, sampleMean: Double, populationMean: Double) -> Double {
    return (pow(z, 2.0) * pow(stdDev, 2.0))/pow((sampleMean - populationMean), 2.0)
}

public func requiredSampleSize(ci: Double, stdDev: Double, sampleMean: Double, populationMean: Double) -> Double {
    let z = zScore(ci: ci)
    return requiredSampleSize(z: z, stdDev: stdDev, sampleMean: sampleMean, populationMean: populationMean)
}

public func requiredSampleSizeProb(ci: Double, prob: Double, maxError: Double) -> Double {
    let z = zScore(ci: ci)
    return (pow(z, 2.0) * prob * (1 - prob))/(pow(maxError, 2.0))
}

//Displaying human readable time format – Solution 3 from https://izziswift.com/how-to-format-time-intervals-for-user-display-social-network-like-in-swift/
extension TimeInterval {
    public func format(using units: NSCalendar.Unit) -> String? {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = units
        formatter.unitsStyle = .abbreviated
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: self)
    }
    public func formatted() -> String? {
        return self.format(using: [.hour, .minute, .second])
    }
}

public struct Tests {
    // A very simplified order class, used as a data source to verify calculations
    public class Order: Codable, ObservableObject {
        let vendor: String
        @Published var orderDateString: String
        @Published var shipDateString: String
        let orderItems: Double
        let orderValue: Double
        let formatter = DateFormatter()
        let calendar = Calendar.current
        var orderDate: Date { formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZZZZZ"; return formatter.date(from: orderDateString) ?? Date() }
        var shipDate: Date { formatter.dateFormat = "yyyy-MM-dd'T'HH:mm:ssZZZZZ"; return formatter.date(from: shipDateString) ?? Date() }
        var orderDateComponents: DateComponents { calendar.dateComponents([Calendar.Component.year, Calendar.Component.month, Calendar.Component.day], from: self.orderDate) }
        var shipDateComponents: DateComponents { calendar.dateComponents([Calendar.Component.year, Calendar.Component.month, Calendar.Component.day], from: self.shipDate) }
        
        enum CodingKeys: String, CodingKey {
            case vendor
            case orderDateString
            case shipDateString
            case orderItems
            case orderValue
        }
        
        required public init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            self.vendor = try values.decode(String.self, forKey: .vendor)
            self.orderDateString = try values.decode(String.self, forKey: .orderDateString)
            self.shipDateString = try values.decode(String.self, forKey: .shipDateString)
            self.orderItems = try values.decode(Double.self, forKey: .orderItems)
            self.orderValue = try values.decode(Double.self, forKey: .orderValue)
        }
        
        public func encode(to encoder: Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(vendor, forKey: .vendor)
            try container.encode(orderDateString, forKey: .orderDateString)
            try container.encode(shipDateString, forKey: .shipDateString)
            try container.encode(orderItems, forKey: .orderItems)
            try container.encode(orderValue, forKey: .orderValue)
        }
        
        func shippingTime() -> TimeInterval {
            return self.shipDate.timeIntervalSinceReferenceDate - self.orderDate.timeIntervalSinceReferenceDate
        }
    }

    public class OrderStore: Codable, ObservableObject {
        @Published var orders: [Order]
        
        init(_ orders: [Order]) {
            self.orders = orders
        }
        
        enum CodingKeys: String, CodingKey {
            case orders
        }
        
        required public init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: CodingKeys.self)
            self.orders = try values.decode([Order].self, forKey: .orders)
        }
        
        public func encode(to encoder:Encoder) throws {
            var container = encoder.container(keyedBy: CodingKeys.self)
            try container.encode(orders, forKey: .orders)
        }
        
        //Filtering orders against a particular vendors
        func filter(_ vendor: String) -> [Order] {
            return self.orders.filter({$0.vendor == vendor})
        }
    }

    public func sampleTests() {
        //Link to sampleOrders.json
        let orderDataURL = Bundle.main.url(forResource: "sampleOrders", withExtension: "json")
        let orderData = try! Data(contentsOf: orderDataURL!)

        let decoder = JSONDecoder()
        let orderStore = try! decoder.decode(OrderStore.self, from: orderData)
        let vendor = "Vendor 1"

        //Should be 1600 orders
        print("Order Count: \(orderStore.orders.count) orders")
        let orderValues = orderStore.orders.map({$0.orderValue})
        //Average Order Value for all orders should be about 54.484
        print("Average Order Value: \(average(orderValues))")
        //Standard Deviation of Average Order Value should be about 26.038
        print("Standard Dev of AOV: \(stdDev(orderValues))")

        //For Vendor 1, AOV should be about 53.021
        print("AOV for \(vendor): \(average(orderStore.orders.filter({$0.vendor == vendor}).map({$0.orderValue})))")
        //For Vendor 1, AOV should be about 26.891
        print("Std Dev of AOV for \(vendor): \(stdDev(orderStore.orders.filter({$0.vendor == vendor}).map({$0.orderValue}), .sample))")
        //For Vendor 1 should range from 0.315 to 105.727
        print("Confidence Interval: \(confidenceInterval(ci: 0.95, values: orderStore.orders.filter({$0.vendor == vendor}).map({$0.orderValue})))")

        
        let shippingTimes = orderStore.orders.map({$0.shippingTime()})
        let shippingDesc = descriptives(shippingTimes)

        //Unimportant tests below
        let vendorNumbers = Array(0..<10)
        let vendors =  vendorNumbers.map({"Vendor \($0)"})
//        let months = Array(1...12).map({$0})
//        let years = [2018, 2019, 2020]

        print("Average Order Values:")
        vendors.map({print("\($0): \(descriptives(orderStore.filter($0).map({$0.orderValue})))")})

        let shippingTimesByVendor = vendors.map({descriptives(orderStore.filter($0).map({$0.shippingTime()}))})
        print("Shipping Times:")
        shippingTimesByVendor.map({print("µ: \($0.mean.formatted() ?? "")\t∂: \($0.stdDev.formatted() ?? "")")})

        print("Mean Shipping Time: \(shippingDesc.mean.format(using: [.hour, .minute, .second]) ?? "")")
        print("Std Dev Shipping Time: \(shippingDesc.stdDev.format(using: [.hour, .minute, .second]) ?? "")")
    }
}
Tests().sampleTests()
