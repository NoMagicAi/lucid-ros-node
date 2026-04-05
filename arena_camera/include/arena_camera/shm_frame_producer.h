// GPU-accelerated Bayer demosaic + shared memory ring buffer producer.
//
// Demosaics raw Bayer frames on GPU (NPP) and writes the resulting RGB frames
// to a POSIX shared memory ring buffer. Multiple consumers (video_saver, IMF,
// etc.) can read the demosaiced frames without per-consumer demosaic overhead.
//
// Usage:
//   ShmFrameProducer producer;
//   if (producer.init("camera_lucid_wrist_left", 1440, 1080, NPPI_BAYER_RGGB,
//   16)) {
//     // In frame grab loop:
//     producer.write(raw_bayer_data, bayer_size, timestamp_ns, frame_id);
//   }
//   // Destructor handles cleanup.

#ifndef ARENA_CAMERA_SHM_FRAME_PRODUCER_H_
#define ARENA_CAMERA_SHM_FRAME_PRODUCER_H_

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <npp.h>

#include <nomagiclib/shm_frame_ringbuf/shm_ring_buffer.h>

namespace arena_camera {

class ShmFrameProducer {
public:
  ShmFrameProducer() = default;
  ~ShmFrameProducer() { destroy(); }

  ShmFrameProducer(const ShmFrameProducer &) = delete;
  ShmFrameProducer &operator=(const ShmFrameProducer &) = delete;

  /// Initialize the producer: allocate GPU buffers, create shared memory ring
  /// buffer.
  /// @param camera_name  Camera identifier (used in shm name:
  /// "nmgc_cam_ring_{camera_name}").
  /// @param width        Frame width in pixels.
  /// @param height       Frame height in pixels.
  /// @param bayer_pattern NPP Bayer pattern (e.g., NPPI_BAYER_RGGB).
  /// @param num_slots    Number of ring buffer slots.
  /// @return true on success.
  bool init(const std::string &camera_name, size_t width, size_t height,
            NppiBayerGridPosition bayer_pattern, uint32_t num_slots) {
    if (initialized_) {
      fprintf(stderr, "ShmFrameProducer: already initialized\n");
      return false;
    }

    width_ = width;
    height_ = height;
    bayer_pattern_ = bayer_pattern;

    // Allocate GPU buffer for single-channel Bayer input
    cudaError_t err =
        cudaMallocPitch(&d_bayer_, &d_bayer_pitch_, width, height);
    if (err != cudaSuccess) {
      fprintf(stderr,
              "ShmFrameProducer: cudaMallocPitch for bayer failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }

    // Allocate GPU buffer for 3-channel RGB output
    err = cudaMallocPitch(&d_rgb_, &d_rgb_pitch_, width * 3, height);
    if (err != cudaSuccess) {
      fprintf(stderr, "ShmFrameProducer: cudaMallocPitch for rgb failed: %s\n",
              cudaGetErrorString(err));
      cudaFree(d_bayer_);
      d_bayer_ = nullptr;
      return false;
    }

    // Allocate CPU staging buffer for RGB output
    rgb_staging_.resize(width * height * 3);

    // Initialize NPP stream context
    if (!initNppStreamContext()) {
      freeGpuBuffers();
      return false;
    }

    // Create shared memory ring buffer
    std::string shm_name = "nmgc_cam_ring_" + camera_name;
    if (!producer_.create(shm_name, static_cast<uint32_t>(width),
                          static_cast<uint32_t>(height), "rgb8", 3,
                          num_slots)) {
      fprintf(stderr, "ShmFrameProducer: failed to create ring buffer '%s'\n",
              shm_name.c_str());
      freeGpuBuffers();
      return false;
    }

    initialized_ = true;
    return true;
  }

  /// Write a Bayer frame: upload to GPU, demosaic to RGB, download, write to
  /// ring buffer.
  /// @param raw_bayer    Pointer to raw Bayer data (host memory).
  /// @param bayer_size   Size of Bayer data in bytes (must equal width *
  /// height).
  /// @param timestamp_ns Frame timestamp in nanoseconds.
  /// @param frame_id     Camera-assigned frame ID.
  /// @return true on success.
  bool write(const uint8_t *raw_bayer, size_t bayer_size, uint64_t timestamp_ns,
             uint64_t frame_id) {
    if (!initialized_) {
      return false;
    }

    // Upload Bayer to GPU
    cudaError_t err = cudaMemcpy2D(d_bayer_, d_bayer_pitch_, raw_bayer, width_,
                                   width_, height_, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "ShmFrameProducer: bayer upload failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }

    // NPP demosaic: single-channel Bayer -> 3-channel interleaved RGB
    NppiSize src_size = {static_cast<int>(width_), static_cast<int>(height_)};
    NppiRect src_roi = {0, 0, static_cast<int>(width_),
                        static_cast<int>(height_)};
    NppStatus npp_status = nppiCFAToRGB_8u_C1C3R_Ctx(
        static_cast<const Npp8u *>(d_bayer_), static_cast<int>(d_bayer_pitch_),
        src_size, src_roi, static_cast<Npp8u *>(d_rgb_),
        static_cast<int>(d_rgb_pitch_), bayer_pattern_, NPPI_INTER_UNDEFINED,
        npp_ctx_);
    if (npp_status != NPP_SUCCESS) {
      fprintf(stderr, "ShmFrameProducer: NPP demosaic failed with status %d\n",
              npp_status);
      return false;
    }

    // Download RGB from GPU to staging buffer
    err = cudaMemcpy2D(rgb_staging_.data(), width_ * 3, d_rgb_, d_rgb_pitch_,
                       width_ * 3, height_, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
      fprintf(stderr, "ShmFrameProducer: rgb download failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }

    // Write to ring buffer
    return producer_.write_frame(rgb_staging_.data(), rgb_staging_.size(),
                                 timestamp_ns, frame_id);
  }

  /// Clean up GPU buffers and shared memory.
  void destroy() {
    if (!initialized_) {
      return;
    }
    freeGpuBuffers();
    producer_.destroy();
    initialized_ = false;
  }

  bool is_initialized() const { return initialized_; }

private:
  bool initNppStreamContext() {
    int device_id = 0;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
      fprintf(stderr, "ShmFrameProducer: cudaGetDevice failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }

    memset(&npp_ctx_, 0, sizeof(npp_ctx_));
    npp_ctx_.hStream = nullptr; // Default CUDA stream

    cudaDeviceProp props;
    err = cudaGetDeviceProperties(&props, device_id);
    if (err != cudaSuccess) {
      fprintf(stderr, "ShmFrameProducer: cudaGetDeviceProperties failed: %s\n",
              cudaGetErrorString(err));
      return false;
    }

    npp_ctx_.nCudaDeviceId = device_id;
    npp_ctx_.nMultiProcessorCount = props.multiProcessorCount;
    npp_ctx_.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    npp_ctx_.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
    npp_ctx_.nSharedMemPerBlock = props.sharedMemPerBlock;

    cudaDeviceGetAttribute(&npp_ctx_.nCudaDevAttrComputeCapabilityMajor,
                           cudaDevAttrComputeCapabilityMajor, device_id);
    cudaDeviceGetAttribute(&npp_ctx_.nCudaDevAttrComputeCapabilityMinor,
                           cudaDevAttrComputeCapabilityMinor, device_id);

    return true;
  }

  void freeGpuBuffers() {
    if (d_bayer_) {
      cudaFree(d_bayer_);
      d_bayer_ = nullptr;
    }
    if (d_rgb_) {
      cudaFree(d_rgb_);
      d_rgb_ = nullptr;
    }
  }

  bool initialized_{false};
  size_t width_{0};
  size_t height_{0};
  NppiBayerGridPosition bayer_pattern_{NPPI_BAYER_RGGB};

  // GPU buffers
  void *d_bayer_{nullptr};
  size_t d_bayer_pitch_{0};
  void *d_rgb_{nullptr};
  size_t d_rgb_pitch_{0};

  // CPU staging buffer for demosaiced RGB
  std::vector<uint8_t> rgb_staging_;

  // NPP stream context
  NppStreamContext npp_ctx_{};

  // Shared memory ring buffer producer
  nomagiclib::ShmRingProducer producer_;
};

} // namespace arena_camera

#endif // ARENA_CAMERA_SHM_FRAME_PRODUCER_H_
