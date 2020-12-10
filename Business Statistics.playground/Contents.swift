import UIKit
import Foundation

protocol Numeric: Comparable {}

extension Float: Numeric {}
extension Double: Numeric {}
extension Int: Numeric {}

// Ranking elements in an array – not critical for this project
extension Array where Element: Numeric {
    
    func rank() -> [Double] {
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
    
    func reverseRank() -> [Double] {
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
    
    func tauAdjustment() -> Double {
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

// MARK – Basics – Mean and Standard Deviation calculations
// We default to using Doubles
public func mean(_ x: [Double]) -> Double {
    guard x.count > 0 else {
        return 0
    }
    return (x.reduce(0.0, +) / Double(x.count))
}

public func average(_ x: [Double]) -> Double {
    return mean(x)
}

func median(_ values: [Double]) -> Double {
    guard values.count > 0 else {
        return 0
    }
    return values.reduce(0, +) / Double(values.count)
}

func sumOfSquaredAvgDiff(_ values: [Double]) -> Double {
    return values.map{ pow($0 - median(values), 2.0)}.reduce(0, {$0 + $1})
}

// Standard deviation for a population, used when you have all observations of a set
func stdDevP(_ values: [Double]) -> Double {
    return sqrt(sumOfSquaredAvgDiff(values)/Double(values.count))
}

// Standard deviation for a sample, used when you do not have all observations of the population
func stdDevS(_ values: [Double]) -> Double {
    return sqrt(sumOfSquaredAvgDiff(values)/Double(values.count - 1))
}

// Default to using sample, which is more likely, but enable population when available
enum Population: String {
    case population
    case sample
}

func stdDev(_ values: [Double], _ pop: Population = .sample) -> Double {
    switch pop {
        case .population:
            return stdDevP(values)
        default:
            return stdDevS(values)
    }
}

//Function to return a tuple of mean and standard deviation for a set of values
func descriptives(_ values: [Double]) -> (mean: Double, stdDev: Double) {
    return (average(values), stdDev(values))
}

// Spearman's Rho is not ciritcal
func spearmansRho(_ independent: [Double], vs variable: [Double]) -> Double {
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

func percentile(zScore z: Double) -> Double {
    return 0.5 * (1 + erf(z / sqrt(2.0)))
}

// Probability Distribution Functon
func normalPDF(x: Double, mean: Double = 0.0, stdDev: Double = 1.0) -> Double {
    let sqrt2Pi = sqrt(2 * Double.pi)
    let xMinusMeanSquared = (x - mean) * (x - mean)
    let stdDevSquaredTimesTwo = 2 * stdDev * stdDev
    let numerator = exp(-xMinusMeanSquared / stdDevSquaredTimesTwo)
    return numerator / (sqrt2Pi * stdDev)
}

// Normal Cumulative Distribution Function
func normalCDF(x: Double, mean: Double = 0.0, stdDev: Double = 1.0) -> Double {
    return (1 + erf((x - mean) / sqrt(2.0) / stdDev)) / 2
}

func inverseNormalCDF(p: Double, mean: Double = 0.0, stdDev: Double = 1.0, tolerance: Double = 0.00001) -> Double {
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

func uniformCDF(x: Double) -> Double {
    if x < 0 {
        return 0.0
    }
    else if x < 1 {
        return x
    }
    
    return 1.0
}

func bernoulliTrial(p: Double) -> Int {
    if drand48() < p {
        return 1
    }
    return 0
}

func binomial(n: Int, p: Double) -> Int {
    var sum = 0
    for _ in 0..<n {
        sum += bernoulliTrial(p: p)
    }
    return sum
}

func confidenceInterval(ci: Double, range: [Double]) -> (low: Double, high: Double) {
    let lowProb = (1 - ci) / 2
    let highProb = 1 - lowProb
    
    let lowValue = inverseNormalCDF(p: lowProb, mean: median(range), stdDev: stdDev(range))
    let highValue = inverseNormalCDF(p: highProb, mean: median(range), stdDev: stdDev(range))
    
    return (lowValue, highValue)
}

// What we are trying to get to. Understanding if a single observation is above or below our particular confidence interval for a given range
func interestingObservation(observation x: Double, range: [Double], confidenceInterval ci: Double) -> Bool {
    let ciRange = confidenceInterval(ci: ci, range: range)
    if x <= ciRange.low || x >= ciRange.high {
        return true
    }
    return false
}

func covariancePopulation(x: [Double], y: [Double]) -> Double {
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

func factorial(_ N: Int) -> Int {
    if N < 1 { return 1 }
    var result = 1
    for i in 1...N {
        result = i * result
    }
    return result
}

func chi2cdf(x: Double, dF: Double, slices: Double = 1000) -> Double {
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

func sampleCorrelationCoefficient(_ independent: [Double], vs variable: [Double]) -> Double {
    let numerator = covariancePopulation(x: independent, y: variable)
    let denominator = (stdDev(independent) * stdDev(variable))
    let r = numerator / denominator
    return r
}

func fisher(_ r: Double) -> Double {
    return (log((1 + r) / (1 - r)) / 2)
}

func tStatistic(_ rho: Double, dFr: Double) -> Double {
    let tStatistic = rho * sqrt(dFr / (1 - (rho *  rho)))
    return tStatistic
}

func tStatistic(_ independent: [Double], _ variable: [Double]) -> Double {
    return tStatistic(spearmansRho(independent, vs: variable), dFr: Double(independent.count - 2))
}

func pValueStudent(_ tValue: Double, dFr: Double) -> Double {
    let rhoTop = tgamma((dFr + 1) / 2.0)
    let rhoBot = sqrt(dFr * Double.pi) * tgamma(dFr / 2)
    let left = rhoTop / rhoBot
    let center = (1 + ((tValue * tValue)/dFr))
    let centEx = -1 * ((dFr + 1) / 2)
    let right = pow(center, centEx)
    let pValueStudent = left * right
    return pValueStudent
}

func pValue(_ independent: [Double], _ variable: [Double]) -> Double {
    return pValueStudent(tStatistic(independent, variable), dFr: Double(independent.count - 2))
}

func derivativeOf(_ fn: (Double) -> Double, at x: Double) -> Double {
    let h: Double = 0.0000001
    return (fn(x + h) - fn(x) / h)
}

func zScore(rho: Double, items: Int) -> Double {
    return sqrt(Double(items - 3)/1.06) * fisher(rho)
}

func zScore(fisherR: Double, items: Int) -> Double {
    return sqrt(Double(items - 3)/1.06) * fisherR
}

func rho(from fisherR: Double) -> Double {
    let top = exp(2.0 * fisherR) - 1
    let bottom = exp(2.0 * fisherR) + 1
    return top / bottom
}

func correlationBreakpoint(_ items: Int, probability: Double) -> Double {
    let zComponents = sqrt(Double(items - 3)/1.06)
    let fisherR = inverseNormalCDF(p: probability) / zComponents
    return rho(from: fisherR)
}

// A very simplified order class, used as a data source to verify calculations
class Order: Codable {
    let vendor: String
    let orderDateString: String
    let shipDateString: String
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
    
    required init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.vendor = try values.decode(String.self, forKey: .vendor)
        self.orderDateString = try values.decode(String.self, forKey: .orderDateString)
        self.shipDateString = try values.decode(String.self, forKey: .shipDateString)
        self.orderItems = try values.decode(Double.self, forKey: .orderItems)
        self.orderValue = try values.decode(Double.self, forKey: .orderValue)
    }
    
    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(vendor, forKey: .vendor)
        try container.encode(orderDateString, forKey: .orderDateString)
        try container.encode(shipDateString, forKey: .shipDateString)
        try container.encode(orderItems, forKey: .orderItems)
        try container.encode(orderValue, forKey: .orderValue)
    }
}

class OrderStore: Codable {
    let orders: [Order]
    
    init(_ orders: [Order]) {
        self.orders = orders
    }
    
    enum CodingKeys: String, CodingKey {
        case orders
    }
    
    required init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        self.orders = try values.decode([Order].self, forKey: .orders)
    }
    
    func encode(to encoder:Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(orders, forKey: .orders)
    }
}

//Link to sampleOrders.json
let orderDataURL = Bundle.main.url(forResource: "sampleOrders", withExtension: "json")
let orderData = try! Data(contentsOf: orderDataURL!)

let decoder = JSONDecoder()
let orderStore = try decoder.decode(OrderStore.self, from: orderData)
let vendor = "Vendor 1"

//Should be 1600 orders
print("Order Count: \(orderStore.orders.count) orders")
//Average Order Value for all orders should be about 54.484
print("Average Order Value: \(average(orderStore.orders.map({$0.orderValue})))")
//Standard Deviation of Average Order Value should be about 26.038
print("Standard Dev of AOV: \(stdDev(orderStore.orders.map({$0.orderValue})))")

//For Vendor 1, AOV should be about 53.021
print("AOV for \(vendor): \(average(orderStore.orders.filter({$0.vendor == vendor}).map({$0.orderValue})))")
//For Vendor 1, AOV should be about 26.891
print("Std Dev of AOV for \(vendor): \(stdDev(orderStore.orders.filter({$0.vendor == vendor}).map({$0.orderValue})))")
//For Vendor 1 should range from 0.315 to 105.727
print("Confidence Interval: \(confidenceInterval(ci: 0.95, range: orderStore.orders.filter({$0.vendor == vendor}).map({$0.orderValue})))")

//Mean Shipping time should be about 74 hours
func shippingTime(_ order: Order) -> TimeInterval {
    return order.shipDate.timeIntervalSinceReferenceDate - order.orderDate.timeIntervalSinceReferenceDate
}
let shippingTimes = orderStore.orders.map({shippingTime($0)})
let shippingDesc = descriptives(shippingTimes)

//Displaying human readable time format – Solution 3 from https://izziswift.com/how-to-format-time-intervals-for-user-display-social-network-like-in-swift/
extension TimeInterval {
    func format(using units: NSCalendar.Unit) -> String? {
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = units
        formatter.unitsStyle = .abbreviated
        formatter.zeroFormattingBehavior = .pad
        return formatter.string(from: self)
    }
    func formatted() -> String? {
        return self.format(using: [.hour, .minute, .second])
    }
}

//Filtering orders against a particular vendors
func filter(_ vendor: String, ordersStore: OrderStore) -> [Order] {
    return ordersStore.orders.filter({$0.vendor == vendor})
}

//Unimportant tests below
let vendorNumbers = Array(0..<10)
let vendors =  vendorNumbers.map({"Vendor \($0)"})
let months = Array(1...12).map({$0})
let years = [2018, 2019, 2020]

print("Average Order Values:")
vendors.map({print("\($0): \(descriptives(filter($0, ordersStore: orderStore).map({$0.orderValue})))")})

let shippingTimesByVendor = vendors.map({descriptives(filter($0, ordersStore: orderStore).map({shippingTime($0)}))})
print("Shipping Times:")
shippingTimesByVendor.map({print("µ: \($0.mean.formatted() ?? "")\t∂: \($0.stdDev.formatted() ?? "")")})

print("Mean Shipping Time: \(shippingDesc.mean.format(using: [.hour, .minute, .second]) ?? "")")
print("Std Dev Shipping Time: \(shippingDesc.stdDev.format(using: [.hour, .minute, .second]) ?? "")")

