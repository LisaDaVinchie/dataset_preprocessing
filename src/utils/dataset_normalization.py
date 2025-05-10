import torch as th

class MinMaxNormalization:
    def __init__(self, batch_size: int = 1000, channels: list = None):
        self.min_val = None
        self.max_val = None
        self.batch_size = batch_size  # Default batch size for normalization
        self.channels = channels  # List of channels to normalize, if None normalize all channels
    
    def normalize(self, images: th.Tensor, nans_masks: th.Tensor) -> tuple[th.Tensor, list]:
        """Normalize the dataset using min-max normalization.
        Exlcude from the normalization the masked pixels, leaving them out of the normalization and min and max calculation.
        Does not handle images with NaNs outside the masked pixels.

        Args:
            dataset (th.Tensor): dataset to normalize
            nans_masks (th.Tensor): mask to use for normalization. 0 where the values are masked, 1 where the values are not masked
            channels (list): list of channels to normalize. If None, normalize all channels

        Returns:
            th.Tensor: normalized dataset
            list: min and max values used for normalization
        """
        
        nans_masks = nans_masks.bool()
        n_channels = images.shape[1]
        n_images = images.shape[0]
        
        # Initialize min and max values for each channel
        self.min_val = th.full((n_channels,), th.inf)
        self.max_val = th.full((n_channels,), -th.inf)
        
        if self.channels is None:
            self.channels = list(range(n_channels))  # Normalize all channels if none specified
        
        # For small datasets, normalize the entire dataset at once
        # Faster but requires more memory
        if n_images < self.batch_size:
            # Get the min and max values of the dataset, excluding the masked pixels
            norm_images = th.empty_like(images)
            # Iterate over the specified channels
            for channel in self.channels:
                valid_pixels = images[:, channel, :, :][nans_masks[:, channel, :, :]]
                self.min_val[channel] = th.min(valid_pixels)
                self.max_val[channel] = th.max(valid_pixels)
                # Normalize where the mask is 1, keep the original values where the mask is 0
                norm_images[:, channel, :, :] = th.where(nans_masks[:, channel, :, :], (images[:, channel, :, :] - self.min_val[channel]) / (self.max_val[channel] - self.min_val[channel]), images[:, channel, :, :])
        
        # For larger datasets, normalize in batches
        # More memory efficient but slower
        else:
            
            # Find global min/max in batches
            for i in range(0, n_images, self.batch_size):
                batch_images = images[i:i+self.batch_size]
                batch_masks = nans_masks[i:i+self.batch_size]
                batch_valid = []
                for channel in self.channels:
                    batch_valid.append(batch_images[:, channel, :, :][batch_masks[:, channel, :, :]])  # Smaller temporary tensor

                if len(batch_valid) > 0:  # Avoid empty tensors
                    self.min_val = [min(self.min_val[channel], th.min(batch_valid[channel]).item()) for channel in self.channels]
                    self.max_val = [max(self.max_val[channel], th.max(batch_valid[channel]).item()) for channel in self.channels]
            
            self.max_val = th.tensor(self.max_val)
            self.min_val = th.tensor(self.min_val)
            norm_images = th.empty_like(images)
            
            scale_factor = 1 / (self.max_val - self.min_val)
            for i in range(0, n_images, self.batch_size):
                if i + self.batch_size > n_images:
                    self.batch_size = n_images - i
                    
                batch_images = images[i:i+self.batch_size]
                batch_masks = nans_masks[i:i+self.batch_size]
                # Iterate over the specified channels
                for channel in self.channels:
                    norm_images[i:i+self.batch_size, channel, :, :] = th.where(batch_masks[:, channel, :, :], (batch_images[:, channel, :, :] - self.min_val[channel]) * scale_factor[channel], batch_images[:, channel, :, :])
        
        return norm_images, [self.min_val, self.max_val]
    
    def denormalize(self, images: th.Tensor, minmax: list) -> th.Tensor:
        """Denormalize the dataset using min-max denormalization.

        Args:s
            images (th.Tensor): dataset to denormalize, shape (batch_size, channels, height, width)
            minmax (list): min and max values used for normalization, one per channel

        Returns:
            th.Tensor: denormalized dataset, shape (batch_size, channels, height, width)
        
        Raises:
            ValueError: If min and max values are None
        """
        
        min_val = minmax[0]
        max_val = minmax[1]
            
        if min_val is None or max_val is None:
            raise ValueError("Min and max values must be provided for denormalization.")
        
        if self.channels is None:
            self.channels = list(range(images.shape[1]))
        
        # Denormalize the images
        denorm_images = th.empty_like(images)
        
        diff = max_val - min_val
        
        for channel in self.channels:
            denorm_images[:, channel, :, :] = images[:, channel, :, :] * diff[channel] + min_val[channel]
        
        return denorm_images