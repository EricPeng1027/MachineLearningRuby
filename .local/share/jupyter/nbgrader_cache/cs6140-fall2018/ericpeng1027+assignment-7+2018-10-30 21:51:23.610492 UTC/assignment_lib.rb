require 'test/unit/assertions'
require 'json'
require 'daru'
require 'distribution'
include Test::Unit::Assertions



def spiral_dataset
  u = Array.new
  97.times do |i|
    angle = i * Math::PI / 16.0
    radius = 6.5 * (104 - i) / 104.0
    x = radius * Math.sin(angle)
    y = radius * Math.cos(angle)

    u << {"features" => {"x1" => x, "x2" => y}, "label" => 1.0}
    u << {"features" => {"x1" => -x, "x2" => -y}, "label" => 0.0}    
  end
  return {"data" => u, "features" => ["x1", "x2"], "labels" => ["1", "0"]}
end





module Assignment3
   
class BinomialModel
  def func dataset, w
    mu = w["mu"]
    dataset.inject(0.0) do |u,row| 
      x = row["label"]
      u -= Math.log((mu ** x) * ((1 - mu) ** (1 - x)))
    end
  end
  
  def grad dataset, w
    mu = w["mu"]
    g = Hash.new {|h,k| h[k] = 0.0}
    dataset.each do |row| 
      x = row["label"]
      g["mu"] -= (x / mu) - (1 - x) / (1 - mu) 
    end

    return g
  end
  
  ## Adjusts the parameter to be within the allowable range
  def adjust w
    w["mu"] = [[0.001, w["mu"]].max, 0.999].min
  end
end
    
class NaiveBayesModel
  def func dataset, w
    -dataset.inject(0.0) do |u, row| 
      cls = row["label"].to_f > 0 ? "pos" : "neg"
      p = cls == "pos" ? 1.0 : 0.0      
      u += Math.log((w["pos_bias"] ** p) * ((1 - w["pos_bias"]) ** (1 - p)))
      n = cls == "neg" ? 1.0 : 0.0      
      u += Math.log((w["neg_bias"] ** n) * ((1 - w["neg_bias"]) ** (1 - n)))
      
      u += row["features"].keys.inject(Math.log(w["#{cls}_bias"])) do |u, fname|
        u += Math.log(w["#{cls}_#{fname}"]) * row["features"][fname]
      end
    end
  end
  
  def grad dataset, w
    g = Hash.new {|h,k| h[k] = 0.0}
    dataset.each do |row|       
      cls = row["label"].to_f > 0 ? "pos" : "neg"
      p = cls == "pos" ? 1.0 : 0.0      
      g["pos_bias"] -= (p / w["pos_bias"]) - (1 - p) / (1 - w["pos_bias"])
      
      n = cls == "neg" ? 1.0 : 0.0      
      g["neg_bias"] -= (n / w["neg_bias"]) - (1 - n) / (1 - w["neg_bias"])

      
      row["features"].each_key do |fname|
        g["#{cls}_#{fname}"] -= row["features"][fname] / w["#{cls}_#{fname}"]
      end
    end

    return g
  end
  
  def predict w, row
    scores = Hash.new
    
    %w(pos neg).each do |cls|
      scores[cls] = row["features"].keys.inject(Math.log(w["#{cls}_bias"])) do |u, fname|
        u += Math.log(w["#{cls}_#{fname}"]) * row["features"][fname]
      end
    end
    cls = scores.keys.max_by {|cls| scores[cls]}
    lbl = cls == "pos" ? "1" : "0"
    {lbl => scores[cls]}
  end
  
  def adjust w
    w.each_key do |fname|
      w[fname] = [[0.001, w[fname]].max, 0.999].min
    end
  end
end

def coin_dataset(n)
  header = %w(x)
  p = 0.7743
  dataset = []
  n.times do
    outcome = rand < p ? 1.0 : 0.0
    dataset << {"features" => {"bias" => 1.0}, "label" => outcome}
  end
  return [header, dataset]
end
    
def plot x, y
  Daru::DataFrame.new({x: x, y: y}).plot(type: :line, x: :x, y: :y) do |plot, diagram|
    plot.x_label "X"
    plot.y_label "Y"
  end
end
   
    
    module Assignment5
    class StochasticGradientDescent
      attr_reader :weights
      attr_reader :objective
      def initialize obj, w_0, lr = 0.01
        @objective = obj
        @weights = w_0
        @n = 1.0
        @lr = lr
      end
      def update x
        dw = @objective.grad(x, @weights)
        learning_rate = @lr / Math.sqrt(@n)

        dw.each_key do |k|
          @weights[k] -= learning_rate * dw[k]
        end

        @objective.adjust @weights
        @n += 1.0
      end
    end
    class LinearRegressionModel
      def func dataset, w
        dataset.inject(0.0) do |u,row| 
          y = row["label"]
          x = row["features"]
          y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
          u += 0.5 * (y_hat - y) ** 2.0
        end
      end

      def grad dataset, w
        g = Hash.new {|h,k| h[k] = 0.0}
        dataset.each do |row| 
          y = row["label"]
          x = row["features"]
          y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
          x.each_key do |k|
            g[k] += (y_hat - y) * x[k]
          end
        end

        g.each_key {|k| g[k] /= dataset.size}
        return g
      end

      ## Adjusts the parameter to be within the allowable range
      def adjust w
      end

      def predict row, w
        x = row["features"]    
        y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]} > 0.0 ? 1 : -1
      end
    end
    class LogisticRegressionModel
      def sigmoid x
        1.0 / (1 + Math.exp(-x))
      end

      def func dataset, w
        dataset.inject(0.0) do |u,row| 
          y = row["label"].to_f > 0 ? 1.0 : -1.0
          x = row["features"]
          y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}

          u += Math.log(1 + Math.exp(-y * y_hat))
        end
      end

      def grad dataset, w
        g = Hash.new {|h,k| h[k] = 0.0}
        dataset.each do |row| 
          y = row["label"].to_f > 0 ? 1.0 : 0.0
          x = row["features"]
          y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
          syh = sigmoid(y_hat)
          x.each_key do |k|
            g[k] += (syh - y) * x[k]
          end
        end
        g.each_key {|k| g[k] /= dataset.size}
        return g
      end

      def predict row, w
        x = row["features"]    
        y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
        sigmoid(y_hat)
      end

      ## Adjusts the parameter to be within the allowable range
      def adjust w
      end
    end
    def roc_curve(scores)
      total_neg = scores.inject(0.0) {|u,s| u += (1 - s.last)}
      total_pos = scores.inject(0.0) {|u,s| u += s.last}
      c_neg = 0.0
      c_pos = 0.0
      fp = [0.0]
      tp = [0.0]
      auc = 0.0
      scores.sort_by {|s| -s.first}.each do |s|
        c_neg += 1 if s.last <= 0
        c_pos += 1 if s.last > 0  

        fpr = c_neg / total_neg
        tpr = c_pos / total_pos
        auc += 0.5 * (tpr + tp.last) * (fpr - fp.last)
        fp << fpr
        tp << tpr
      end
      return [fp, tp, auc]
    end

end
end