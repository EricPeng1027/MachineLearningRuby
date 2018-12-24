def create_dataset db, sql
  dataset = [] 
  db.execute sql do |row|
    # BEGIN YOUR CODE
    data = Hash.new
    data["features"] = Hash.new
    
    row.each do |k, v|
      if k == "TARGET"
        data["label"] = row["TARGET"]
      elsif k == "SK_ID_CURR"
        data["id"] = row["SK_ID_CURR"]
      elsif k.is_a? String
        data["features"][k.downcase] = row[k]
      end
    end
    dataset << data
    #END YOUR CODE
  end
  return dataset
end

def class_distribution dataset
  # BEGIN YOUR CODE
  dist = Hash.new{|h, k| h[k] = 0}
  
  sum = 0.0
  for row in dataset
    dist[row["label"]] += 1
    sum += 1
  end
  
  dist.each {|k, v| dist[k] = v / sum}
  return dist
  #END YOUR CODE
end

def entropy dist
  # BEGIN YOUR CODE
  sum = 0.0
  dist.each do |k, v|
    sum += v
  end
  
  if sum == 0
    return 0
  end
  
  entropy = 0.0
  dist.each do |k, v|
    if v != 0
      entropy -= v / sum * Math.log(v / sum)
    end
  end
  return entropy
  #END YOUR CODE
end

def find_split_point_numeric dataset, h0, fname
  # BEGIN YOUR CODE
  ig_max = 0
  t_max = nil

  # Deal with missing value
  x = []
  dataset.each do |r|
    if r["features"][fname] != ""
      x << r
    end
  end
  
  feature_groups = x.group_by {|r| r["features"].fetch(fname, 0.0)}
  counts_right = Hash.new {|h,k| h[k] = 0}
  counts_left = Hash.new {|h,k| h[k] = 0}
  v_left = 0.0
  v_right = x.size.to_f

  feature_groups.each_key do |t|
    counts = Hash.new {|h,k| h[k] = 0}  
    feature_groups[t].each do |r| 
      counts[r["label"]] += 1
      counts_right[r["label"]] += 1
    end
    feature_groups[t] = counts
  end
  
  thresholds = feature_groups.keys.sort
  t = thresholds.shift
  
  feature_groups[t].each_key do |k| 
    counts_left[k] += feature_groups[t][k]
    counts_right[k] -= feature_groups[t][k]
    v_left += feature_groups[t][k]
    v_right -= feature_groups[t][k]
  end
  
  thresholds.each.with_index do |t, i|
    p_left = v_left / x.size
    p_right = v_right / x.size
    
    d_left = Hash.new
    d_right = Hash.new
    counts_left.each_key {|k| d_left[k] = counts_left[k] / v_left}
    counts_right.each_key {|k| d_right[k] = counts_right[k] / v_right}
        
    h_left = entropy(d_left)
    h_right = entropy(d_right)    
    ig = h0 - (p_left * h_left + p_right * h_right)
    if ig > ig_max
      ig_max = ig
      t_max = t
    end

    feature_groups[t].each_key do |k| 
      counts_left[k] += feature_groups[t][k]
      counts_right[k] -= feature_groups[t][k]
      v_left += feature_groups[t][k]
      v_right -= feature_groups[t][k]
    end
  end

  return [t_max, ig_max]
  #END YOUR CODE
end

def information_gain h0, splits
  # BEGIN YOUR CODE
  # Deal with missing value
  splits.delete("")
  
  sum = 0
  splits.each do |k, v|
    sum += v.length
  end
  
  ig = h0
  splits.each do |k, v|
    ig -= (entropy(class_distribution(v))) * v.length / sum
  end
  return ig
  #END YOUR CODE
end

def information_gain_categorical x, h0, fname
  splits = x.group_by {|row| row["features"][fname]}
  ig = information_gain h0, splits
  return ig
end

def extract_features db
  dataset = []
  # BEGIN YOUR CODE
  sql = "select * from application_train"
  dataset_all = create_dataset db, sql
  h0 = entropy(class_distribution(dataset_all))
  
  select_features = []
  dataset_all[0]["features"].each do |k, v|
    ig = 0
    if v.is_a? Numeric
      t, ig = find_split_point_numeric dataset_all, h0, k
    elsif v.is_a? String
      ig = information_gain_categorical dataset_all, h0, k
    end
    
    if ig > 0.005
      select_features << k
    end
  end
  puts select_features
  
  dataset_all.each do |row|
    data = Hash.new
    data["features"] = Hash.new
    data["id"] = row["id"]
    data["label"] = row["label"]
    
    for fname in select_features
      data["features"][fname] = row["features"][fname]
    end
    dataset << data
  end
  # END YOUR CODE
  return dataset
end


# Linear Regression Model
def create_dataset_linear db, sql
  dataset = []
  db.execute sql do |row|
    # BEGIN YOUR CODE
    data = Hash.new
    data["label"] = row["TARGET"]
    data["id"] = row["SK_ID_CURR"]
    data["features"] = Hash.new
    row.each_key do |key|
      if key != "TARGET" and key != "SK_ID_CURR" and key.is_a? String
        data["features"][key.downcase] = row[key]==""? 0.0 : row[key]
      end
      
    end
    dataset << data
    #END YOUR CODE
  end
  return dataset
end

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
    # BEGIN YOUR CODE
    dw = @objective.grad(x, @weights)
    learning_rate = @lr / Math.sqrt(@n)
    
    dw.each_key do |k|
      @weights[k] -= learning_rate * dw[k]
    end

    @objective.adjust @weights
    @n += 1.0
    #END YOUR CODE
  end
end

def train_sgd(obj, w, dataset)
  i = 0
  iters = []
  losses = []
  puts dataset[0]
  
  #Define sgd = StochasticGradientDescent.new obj, w, lr
  # You set the learning rate, lr
  # BEGIN YOUR CODE
  sgd = StochasticGradientDescent.new obj, w, 0.05
  10.times do 
    dataset.each_slice(20) do |batch|    
      sgd.update batch
      iters << i
      losses << obj.func(batch, sgd.weights)
      i += 1
    end
  end
  #END YOUR CODE
  return [sgd, iters, losses]
end

class LinearRegressionModel  
  def predict row, w
    # BEGIN YOUR CODE
    x = row["features"]    
    y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
    #END YOUR CODE
  end
  
  def func data, w
    # BEGIN YOUR CODE
    data.inject(0.0) do |u,row| 
      y = row["label"]
      y_hat = self.predict(row, w)
      u += 0.5 * (y_hat - y) ** 2.0
    end / data.size.to_f
    #END YOUR CODE
  end
  ## Adjusts the parameter to be within the allowable range
  def adjust w
  end
    
  def grad data, w
    # BEGIN YOUR CODE
    g = Hash.new {|h,k| h[k] = 0.0}
    data.each do |row| 
      y = row["label"]
      x = row["features"]
      y_hat = self.predict(row, w)
      x.each_key do |k|
        g[k] += (y_hat - y) * x[k]
      end
    end

    g.each_key {|k| g[k] /= data.size}
    return g
    #END YOUR CODE
  end
end

## Logistic Regression Model
def create_dataset_logistic db, sql
  dataset = []
  means = Hash.new {|h, k| h[k] = 0.0}
  size = 0
  db.execute sql do |row|
    row.each_key do |key|
      if key != "TARGET" and key != "SK_ID_CURR" and key.is_a? String
        if row[key] != "" and row[key] != nil
          means[key] += row[key]
        end
      end
    end
    size += 1
  end
    
  means.each do |k, v|
    means[k] = means[k] / size
  end
    
  db.execute sql do |row|
    # BEGIN YOUR CODE
    data = Hash.new
    data["label"] = row["TARGET"]
    data["id"] = row["SK_ID_CURR"]
    data["features"] = Hash.new
    row.each_key do |key|
      if key != "TARGET" and key != "SK_ID_CURR" and key.is_a? String
        data["features"][key.downcase] = (row[key]=="" || row[key]==nil)? means[key] : row[key]
      end
    end
    dataset << data
    #END YOUR CODE
  end
  return dataset
end

def train_logistic_sgd(obj, w, dataset)
  i = 0
  iters = []
  losses = []
  
  #Define sgd = StochasticGradientDescent.new obj, w, lr
  # You set the learning rate, lr
  # BEGIN YOUR CODE
  sgd = StochasticGradientDescent.new obj, w, 1.0
  1.times do 
    dataset.each_slice(20) do |batch|    
      sgd.update batch
      iters << i
      losses << obj.func(batch, sgd.weights)
      i += 1
    end
  end
  #END YOUR CODE
  return [sgd, iters, losses]
end

class LogisticRegressionModel
  def predict row, w
    # BEGIN YOUR CODE
    x = row["features"]    
    yhat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
    1.0 / (1 + Math.exp(-yhat))
    #END YOUR CODE
  end
  
  def adjust w
    w
  end
    
  def func data, w
    # BEGIN YOUR CODE
    data.inject(0.0) do |u,row| 
      y = row["label"].to_f > 0 ? 1.0 : -1.0
      x = row["features"]
      y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
      
      u += Math.log(1 + Math.exp(-y * y_hat))
    end / data.size.to_f
    #END YOUR CODE
  end
    
  def grad data, w
    # BEGIN YOUR CODE
    g = Hash.new {|h,k| h[k] = 0.0}
    data.each do |row| 
      y = row["label"].to_f > 0 ? 1.0 : 0.0
      x = row["features"]
      y_hat = x.keys.inject(0.0) {|s, k| s += w[k] * x[k]}
      syh = 1.0 / (1 + Math.exp(-y_hat))
      x.each_key do |k|
        g[k] += (syh - y) * x[k]
      end
    end
    g.each_key {|k| g[k] /= data.size}
    return g
    #END YOUR CODE
  end
end

## Add mean and stdev here
# BEGIN YOUR CODE
def mean x
  sum = x.inject(0.0) {|u,v| u += v}
  sum / x.size
end

def stdev x
  m = mean x
  sum = x.inject(0.0) {|u,v| u += (v - m) ** 2.0}
  Math.sqrt(sum / (x.size - 1))
end
#END YOUR CODE

def create_zdataset dataset
  zdataset = dataset.clone;
  zdataset = dataset.collect do |r|
    u = r.clone
    u["features"] = r["features"].clone
    u
  end
  puts zdataset[0]["features"].keys
  # BEGIN YOUR CODE
  means = Hash.new
  stdevs = Hash.new
  zdataset[0]["features"].keys.each do |f|
    rs = zdataset.select {|r| r["features"].has_key? f}
    x = rs.collect {|r| r["features"][f]}
    means[f] = mean x
    stdevs[f] = stdev x    
    rs.each {|r| r["features"][f] = (r["features"][f] - means[f]) / stdevs[f]}
  end
  #END YOUR CODE
  return zdataset
end


## Decision tree
class NumericSplitter
  def new_split dataset, initial_entropy, fname
    # BEGIN YOUR CODE
    x = dataset
    h0 = initial_entropy
    ig_max = 0
    t_max = nil

    feature_groups = x.group_by {|r| r["features"].fetch(fname, 0.0)}
    counts_right = Hash.new {|h,k| h[k] = 0}
    counts_left = Hash.new {|h,k| h[k] = 0}
    v_left = 0.0
    v_right = x.size.to_f

    feature_groups.each_key do |t|
      counts = Hash.new {|h,k| h[k] = 0}  
      feature_groups[t].each do |r| 
        counts[r["label"]] += 1
        counts_right[r["label"]] += 1
      end
      feature_groups[t] = counts
    end

    thresholds = feature_groups.keys.sort
    t = thresholds.shift

    feature_groups[t].each_key do |k| 
      counts_left[k] += feature_groups[t][k]
      counts_right[k] -= feature_groups[t][k]
      v_left += feature_groups[t][k]
      v_right -= feature_groups[t][k]
    end

    thresholds.each.with_index do |t, i|
      p_left = v_left / x.size
      p_right = v_right / x.size

      d_left = Hash.new
      d_right = Hash.new
      counts_left.each_key {|k| d_left[k] = counts_left[k] / v_left}
      counts_right.each_key {|k| d_right[k] = counts_right[k] / v_right}

      h_left = entropy(d_left)
      h_right = entropy(d_right)    
      ig = h0 - (p_left * h_left + p_right * h_right)
      if ig > ig_max
        ig_max = ig
        t_max = t
      end

      feature_groups[t].each_key do |k| 
        counts_left[k] += feature_groups[t][k]
        counts_right[k] -= feature_groups[t][k]
        v_left += feature_groups[t][k]
        v_right -= feature_groups[t][k]
      end
    end

    
    return [nil, 0] if t_max.nil?
    #END YOUR CODE
    [NumericSplit.new(fname, t_max), ig_max]
  end
  
  def matches? x, fname
    x.all? {|r| r["features"].fetch(fname, 0.0).is_a?(Numeric)}
  end
end

class DecisionTree
  attr_reader :tree, :h0
  
  def initialize splitters, min_size, max_depth
    @splitters = splitters
    @min_size = min_size
    @max_depth = max_depth
  end
  
  def init_dataset dataset
    @dataset = dataset
    @header = @dataset["features"]
    @c_dist = class_distribution @dataset["data"]
    @h0 = entropy @c_dist
    @tree = {n: @dataset["data"].size, entropy: @h0, dist: @c_dist, split: nil, children: {}}    
  end
  
  def find_best_split dataset, initial_entropy
    # BEGIN YOUR CODE
    t_max = nil
    ig_max = 0.0
    
    @header.each do |fname|
      split_result = @splitters[0].new_split dataset, initial_entropy, fname
      if split_result[1] > ig_max
        t_max = split_result[0]
        ig_max = split_result[1]
      end
    end
    return [t_max, ig_max]
    #END YOUR CODE
  end

  def train dataset
    init_dataset dataset
    build_tree @dataset["data"], @tree, @max_depth
  end

  def build_tree x, root, max_depth
    # BEGIN YOUR CODE
    if max_depth <= 1 or root[:n] < @min_size
      return
    end
      
    header = @header
    min_size = @min_size
    
    h = root[:entropy]
    best_split, ig = find_best_split x, h 

    return root if best_split.nil?
    root[:split] = best_split
    next_splits = best_split.split x

    children = Hash.new

    next_splits.each_key do |exp|
      dataset = next_splits[exp]
      class_dist = class_distribution(dataset)
      ent = entropy class_dist
      child = {n: dataset.size, entropy: ent, dist: class_dist, split: nil, children: {}}
      children[exp] = child
      build_tree dataset, child, max_depth - 1 
    end
    root[:children] = children

    return root
    #END YOUR CODE
  end

  def predict x
    return eval_tree x, @tree
  end
  
  def eval_tree x, root
    # BEGIN YOUR CODE
    if root[:children].empty?
      return root[:dist].max_by {|k, v| v}[0]
    else
      return eval_tree x, root[:children][root[:split].test(x)]
    end
    #END YOUR CODE
  end
end