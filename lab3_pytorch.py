import torch.nn as nn
from torch.nn.utils.rnn import  pack_padded_sequence, pad_packed_sequence
import torch
from torch.autograd import Variable
import random

class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_size, output_size):
        """Define layers for a vanilla rnn encoder"""
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.gru = nn.GRU(embedding_size, output_size)

    def forward(self, input_seqs, input_lengths, hidden=None):
        embedded = self.embedding(input_seqs)
        packed = pack_padded_sequence(embedded, input_lengths)
        packed_outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = pad_packed_sequence(packed_outputs)
        return outputs, hidden


class Decoder(nn.Module):

   def __init__(self, hidden_size, output_size, max_length, teacher_forcing_ratio, sos_id, use_cuda):
      """Define layers for a vanilla rnn decoder"""
      super(Decoder, self).__init__()

      self.hidden_size = hidden_size
      self.output_size = output_size
      self.embedding = nn.Embedding(output_size, hidden_size)
      self.gru = nn.GRU(hidden_size, hidden_size)
      self.out = nn.Linear(hidden_size, output_size)
      self.log_softmax = nn.LogSoftmax()  # work with NLLLoss = CrossEntropyLoss

      self.max_length = max_length
      self.teacher_forcing_ratio = teacher_forcing_ratio
      self.sos_id = sos_id
      self.use_cuda = use_cuda

   def forward_step(self, inputs, hidden):
      # inputs: (time_steps=1, batch_size)
      batch_size = inputs.size(1)
      embedded = self.embedding(inputs)
      embedded.view(1, batch_size, self.hidden_size)  # S = T(1) x B x N
      rnn_output, hidden = self.gru(embedded, hidden)  # S = T(1) x B x H
      rnn_output = rnn_output.squeeze(0)  # squeeze the time dimension
      output = self.log_softmax(self.out(rnn_output))  # S = B x O
      return output, hidden

   def forward(self, context_vector, targets):

      # Prepare variable for decoder on time_step_0
      target_vars, target_lengths = targets
      batch_size = context_vector.size(1)
      decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))

      # Pass the context vector
      decoder_hidden = context_vector

      max_target_length = max(target_lengths)
      decoder_outputs = Variable(torch.zeros(
         max_target_length,
         batch_size,
         self.output_size
      ))  # (time_steps, batch_size, vocab_size)

      if self.use_cuda:
         decoder_input = decoder_input.cuda()
         decoder_outputs = decoder_outputs.cuda()

      use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False

      # Unfold the decoder RNN on the time dimension
      for t in range(max_target_length):
         decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
         decoder_outputs[t] = decoder_outputs_on_t
         if use_teacher_forcing:
            decoder_input = target_vars[t].unsqueeze(0)
         else:
            decoder_input = self._decode_to_index(decoder_outputs_on_t)
         return decoder_outputs, decoder_hidden

if __name__ == '__main__':
   batch_size = 32  # Batch size for training.
   epochs = 50  # Number of epochs to train for.
   latent_dim = 256  # Latent dimensionality of the encoding space.
   num_samples = 1000000  # Number of samples to train on.
   embedding_dim = 300
   # Path to the data txt file on disk.
   data_path = 'data/train.txt'
   val_data_path = 'data/validation.txt'