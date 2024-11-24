import tensorflow as tf
import numpy as np

concat = tf.keras.backend.concatenate
stack = tf.keras.backend.stack
K = tf.keras.backend
Dense = tf.keras.layers.Dense
Add = tf.keras.layers.Add
LayerNorm = tf.keras.layers.LayerNormalization
Multiply = tf.keras.layers.Multiply
Dropout = tf.keras.layers.Dropout
Activation = tf.keras.layers.Activation
Lambda = tf.keras.layers.Lambda

class MyCustomPrintMaskLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyCustomPrintMaskLayer, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True
        
    def call(self, inputs, mask=None):
        # using 'mask' you can access the mask passed from the previous layer
        tf.print(mask)
        return inputs
    
    
class MyRealEmbedderLayer(tf.keras.layers.Layer):
    def __init__(self, dynamic_size, num_outputs, **kwargs):
        super(MyRealEmbedderLayer, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True
        self.dynamic_size = dynamic_size
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.fc = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.num_outputs))
        self.fc.build(input_shape)
        self._trainable_weights = self.fc.trainable_weights
        super(MyRealEmbedderLayer, self).build(input_shape)
        
    def call(self, input, mask=None):
        # using 'mask' you can access the mask passed from the previous layer
        output = tf.keras.backend.stack(
            [self.fc(input) for i in range(self.dynamic_size)],
            axis=-1
        )
        return output
    
    
class MyReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, time_steps, embedding_dim, num_inputs, **kwargs):
        super(MyReshapeLayer, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True
        self.time_steps = time_steps
        self.embedding_dim = embedding_dim
        self.num_inputs = num_inputs
        
    def call(self, input, mask=None):
        # using 'mask' you can access the mask passed from the previous layer
        output = K.reshape(
            input,
            [-1, self.time_steps, self.embedding_dim * self.num_inputs]
        )
        return output

    
class MyExpandDimsLayer(tf.keras.layers.Layer):
    def __init__(self, axis, **kwargs):
        super(MyExpandDimsLayer, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True
        self.axis = axis

    def call(self, input, mask=None):
        # using 'mask' you can access the mask passed from the previous layer
        output = tf.expand_dims(input, axis=self.axis)
        return output
    
    
class MyStackLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyStackLayer, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True
        
    def call(self, input, mask=None):
        # using 'mask' you can access the mask passed from the previous layer
        output = tf.keras.backend.stack(input, axis=-1)
        mask = mask[0]
        return output

    
class MySplitterLayer(tf.keras.layers.Layer):
    def __init__(self,  index, **kwargs):
        super(MySplitterLayer, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True
        self.index = index

    @tf.function
    def call(self, input, mask=None):
        inputSplitted = Lambda(lambda x: x[Ellipsis,self.index])(input)
        return inputSplitted
    
class MyCustomsSumMasked(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MyCustomsSumMasked, self).__init__(**kwargs)
        # Signal that the layer is safe for mask propagation
        self.supports_masking = True
        
    def call(self, input, mask=None):
        # using 'mask' you can access the mask passed from the previous layer
        mask_4d = tf.transpose((tf.ones([1, 1, 1, 1]) * tf.cast(mask, "float")), perm=[2, 3, 0, 1])
        mask_4d = (mask_4d == 1)
        input_masked = tf.where(mask_4d, input,  tf.zeros_like(input))
        return tf.keras.backend.sum(input_masked, axis=-1)
    

class MyCustomAddLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # I've added pass because this is the simplest form I can come up with.
        self.supports_masking = True

    def call(self, inputs, mask=None):
        input_0, input_1 = inputs
        mask_0, mask_1 = mask
        mask_matrix_0 = tf.transpose((tf.ones([input_0.shape[2], 1, 1]) * tf.cast(mask_0, "float")), perm=[1, 2, 0])

        mask_matrix_0 = (mask_matrix_0 == 1)
        input_0_masked = tf.where(mask_matrix_0, input_0,  tf.zeros_like(input_0))
        
        mask_matrix_1 = tf.transpose((tf.ones([input_1.shape[2], 1, 1]) * tf.cast(mask_1, "float")), perm=[1, 2, 0])
        mask_matrix_1 = (mask_matrix_1 == 1)
        input_1_masked = tf.where(mask_matrix_1, input_1,  tf.zeros_like(input_1))
        
        tmp = tf.keras.layers.Add()([input_0_masked, input_1_masked])
#         tmp = tf.keras.layers.LayerNormalization()(tmp)
        return tmp

class MyCustomAddLayerV2(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # I've added pass because this is the simplest form I can come up with.
        self.supports_masking = True

    def call(self, inputs, mask=None):
        input_0, input_1 = inputs
        mask_0, mask_1 = mask
        
        if mask == None:
            tf.print("Está pasando algo raro..")
            tmp = tf.keras.layers.Add()([input_0, input_1])
        else:
            mask_matrix_0 = tf.transpose((tf.ones([input_0.shape[2], 1, 1]) * tf.cast(mask_0, "float")), perm=[1, 2, 0])
            mask_matrix_0 = (mask_matrix_0 == 1)
            input_0_masked = tf.where(mask_matrix_0, input_0,  tf.zeros_like(input_0))

            mask_matrix_1 = tf.transpose((tf.ones([input_1.shape[2], 1, 1]) * tf.cast(mask_0, "float")), perm=[1, 2, 0])
            mask_matrix_1 = (mask_matrix_1 == 1)
            input_1_masked = tf.where(mask_matrix_1, input_1,  tf.zeros_like(input_1))

            tmp = tf.keras.layers.Add()([input_0_masked, input_1_masked])
        return tmp




# Layer utility functions.
def linear_layer(size,
                 activation=None,
                 use_time_distributed=False,
                 use_bias=True):
    """Returns simple Keras linear layer.
      Args:
        size: Output size
        activation: Activation function to apply if required
        use_time_distributed: Whether to apply layer across time
        use_bias: Whether bias should be included in layer
    """
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear

def gated_residual_network(x,
                           hidden_layer_size,
                           output_size=None,
                           dropout_rate=None,
                           use_time_distributed=True,
                           additional_context=None,
                           return_gate=False):
    """Applies the gated residual network (GRN) as defined in paper.

      Args:
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes

      Returns:
        Tuple of tensors for: (GRN output, GLU gate)
    """
    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    # Apply feedforward network
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size,
            activation=None,
            use_time_distributed=use_time_distributed,
            use_bias=False)(
            additional_context)
    hidden = tf.keras.layers.Activation('elu')(hidden)
    hidden = linear_layer(
        hidden_layer_size,
        activation=None,
        use_time_distributed=use_time_distributed)(
        hidden)

    gating_layer, gate = apply_gating_layer(
        hidden,
        output_size,
        dropout_rate=dropout_rate,
        use_time_distributed=use_time_distributed,
        activation=None)

    if return_gate:
#         if use_time_distributed:
#             return temporal_add_and_norm_v2([skip, gating_layer]), gate
#         else:
        return add_and_norm([skip, gating_layer]), gate
    else:
#         if use_time_distributed:
#             return temporal_add_and_norm_v2([skip, gating_layer])
#         else:
        return add_and_norm([skip, gating_layer])


def apply_gating_layer(x,
                       hidden_layer_size,
                       dropout_rate=None,
                       use_time_distributed=True,
                       activation=None):
    """Applies a Gated Linear Unit (GLU) to an input.

  Args:
    x: Input to gating layer
    hidden_layer_size: Dimension of GLU
    dropout_rate: Dropout rate to apply if any
    use_time_distributed: Whether to apply across time
    activation: Activation function to apply to the linear feature transform if
      necessary

  Returns:
    Tuple of tensors for: (GLU output, gate)
  """

    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation=activation))(
            x)
        gated_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation='sigmoid'))(
            x)
    else:
        activation_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation=activation)(
            x)
        gated_layer = tf.keras.layers.Dense(
            hidden_layer_size, activation='sigmoid')(
            x)

    return tf.keras.layers.Multiply()([activation_layer,
                                       gated_layer]), gated_layer
    
def temporal_add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
    tmp = MyCustomAddLayer()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp
    
def add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
    tmp = Add()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp

def temporal_add_and_norm_v2(x_list):
    """Applies skip connection followed by layer normalisation.

  Args:
    x_list: List of inputs to sum for skip connection

  Returns:
    Tensor output from layer.
  """
    tmp = MyCustomAddLayerV2()(x_list)
    tmp = LayerNorm()(tmp)
    return tmp

def get_TFT_embeddings(static_input,
                       static_size,
                       category_counts,
                       dynamic_input,
                       dynamic_size,
                       hidden_layer_size):

    # define the sizes
    num_categorical_variables = len(category_counts)
    num_regular_variables = static_size - num_categorical_variables

    regular_inputs, categorical_inputs = static_input[:, :num_regular_variables], \
                                         static_input[:, num_regular_variables:]
    embedding_sizes = [hidden_layer_size for i, size in enumerate(category_counts)]

    embedded_inputs = []
    for i in range(num_categorical_variables):
        embedding = tf.keras.layers.Embedding(
            category_counts[i],
            embedding_sizes[i],
            dtype=tf.float32,
            name=("embeddign_" + str(i))
        )(categorical_inputs[Ellipsis, i])
        embedded_inputs.append(embedding)
    static_inputs_tr = tf.keras.backend.stack(
        [tf.keras.layers.Dense(hidden_layer_size)(regular_inputs) for i in range(num_regular_variables)] + \
        [embedded_inputs[i] for i in range(num_categorical_variables)],
        axis=1
    )

    dynamic_inputs_tr = MyRealEmbedderLayer(dynamic_size, hidden_layer_size)(dynamic_input)
    return static_inputs_tr, dynamic_inputs_tr


def static_combine_and_mask(embedding,
                            hidden_layer_size,
                            dropout_rate=None
                            ):
    # Add temporal features
    _, num_static, _ = embedding.get_shape().as_list()

    flatten = tf.keras.layers.Flatten(name="flattened_inputs")(embedding)

    # Nonlinear transformation with gated residual network.
    mlp_outputs = gated_residual_network(
        flatten,
        hidden_layer_size,
        output_size=num_static,
        dropout_rate=dropout_rate,
        use_time_distributed=False,
        additional_context=None)

    sparse_weights = tf.keras.layers.Activation('softmax', name="softmax_act")(mlp_outputs)
    sparse_weights = K.expand_dims(sparse_weights, axis=-1)

    trans_emb_list = []
    for i in range(num_static):
        e = gated_residual_network(
            embedding[:, i:i + 1, :],
            hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=False)
        trans_emb_list.append(e)

    transformed_embedding = concat(trans_emb_list, axis=1)

    combined = tf.keras.layers.Multiply(name="mult")([sparse_weights, transformed_embedding])

    static_vec = K.sum(combined, axis=1)
    
    return static_vec, sparse_weights


def rnn_combine_and_mask(embedding,
                         static_context_variable_selection,
                         hidden_layer_size,
                         dropout_rate=None
                         ):
    """Apply temporal variable selection networks.
        Args:
        embedding: Transformed inputs.

        Returns:
        Processed tensor outputs.
    """
    # Add temporal features

    _, time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()
    
    flatten = MyReshapeLayer(time_steps, embedding_dim, num_inputs)(embedding)
    
    expanded_static_context = K.expand_dims(static_context_variable_selection, axis=1)
    
    # Variable selection weights
    mlp_outputs, static_gate = gated_residual_network(
        flatten,
        hidden_layer_size,
        output_size=num_inputs,
        dropout_rate=dropout_rate,
        use_time_distributed=True,
        additional_context=expanded_static_context,
        return_gate=True)
    
    sparse_weights = tf.keras.layers.Activation('softmax', name="softmax_dyn")(mlp_outputs)
    sparse_weights = MyExpandDimsLayer(axis=2)(sparse_weights)

    # Non-linear Processing & weight application
    trans_emb_list = []
    for i in range(num_inputs):
        embedding_slice = MySplitterLayer(i)(embedding)
        grn_output = gated_residual_network(
            embedding_slice,
            hidden_layer_size,
            dropout_rate=dropout_rate,
            use_time_distributed=True)
        trans_emb_list.append(grn_output)
        
    transformed_embedding = MyStackLayer()(trans_emb_list)

    combined = tf.keras.layers.Multiply()([sparse_weights, transformed_embedding])
    temporal_ctx = MyCustomsSumMasked()(combined)
    
    return temporal_ctx, sparse_weights, static_gate

def get_rnn(hidden_layer_size, return_state):
    """Returns LSTM cell initialized with default parameters."""
    lstm = tf.keras.layers.LSTM(hidden_layer_size,
                                return_sequences=True,
                                return_state=return_state,
                                stateful=False,
                                # Additional params to ensure LSTM matches CuDNN, See TF 2.0 :
                                # (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
                                activation='tanh',
                                recurrent_activation='sigmoid',
                                recurrent_dropout=0,
                                unroll=False,
                                use_bias=True)
    return lstm


def InterpretableMultiHeadAttention(qs, ks, vs, hidden_layer_size, dropout_rate, n_head):
    heads = []
    attns = []
    d_k = d_v = hidden_layer_size // n_head
    for i in range(n_head):
        qs = tf.keras.layers.TimeDistributed(Dense(d_k, use_bias=False))(qs)
        ks = tf.keras.layers.TimeDistributed(Dense(d_k, use_bias=False))(ks)
        vs = tf.keras.layers.TimeDistributed(Dense(d_k, use_bias=False))(vs)
        head, attn = ScaledDotProductAttention()(qs, ks, vs)
        head_dropout = Dropout(dropout_rate)(head)
        heads.append(head_dropout)
        attns.append(attn)

    # head_stacked = K.stack(heads) if n_head > 1 else heads[0]
    attn = K.stack(attns)
    # outputs = K.mean(head, axis=0) if n_head > 1 else head_stacked
    # outputs = self.w_o(outputs)
    # outputs = Dropout(self.dropout)(outputs)  # output dropout
    return head, attn

class ScaledDotProductAttention():
    """Defines scaled dot product attention layer.
      Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g.
          softmax by default)
    """
    def __init__(self, attn_dropout=0.0):
        self.supports_masking = True
        self.dropout = Dropout(attn_dropout)
        self.activation = Activation('softmax')

    def __call__(self, q, k, v, mask=None):
        """Applies scaled dot product attention.

    Args:
      q: Queries
      k: Keys
      v: Values
      mask: Masking if required -- sets softmax to very large value

    Returns:
      Tuple of (layer outputs, attention weights)
    """
        temper = tf.sqrt(tf.cast(tf.shape(k)[-1], dtype='float32'))
        attn = Lambda(lambda x: K.batch_dot(x[0], x[1], axes=[2, 2]))([q, k])  # shape=(batch, q, k)
        attn = attn / temper  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e+9) * (1. - K.cast(x, 'float32')))(mask)  # setting to infinity
            attn = Add()([attn, mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn
