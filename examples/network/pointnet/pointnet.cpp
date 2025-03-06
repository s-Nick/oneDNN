#include <array>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>

#include <oneapi/dnnl/dnnl.hpp>
#include <sycl/sycl.hpp>
#include <unordered_map>

#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_types.h"
//#include "oneapi/dnnl/dnnl_common.hpp"

using DType = float;
static std::array<std::string, 10> object_map = {"bathtub", "bed", "chair",
        "desk", "dresser", "monitor", "night stand", "sofa", "table", "toilet"};

namespace helpers {

template <typename T>
void copy_to_device(std::vector<char> &inputs, T *dev_ptr, sycl::queue q) {
    q.submit([&](sycl::handler &cgh) {
         cgh.copy(inputs.data(), reinterpret_cast<char *>(dev_ptr),
                 inputs.size());
     }).wait_and_throw();
}

// Helper function that reads binary data produced by h5tobin.py into a vector
std::vector<char> read_binary_data(std::string const &name) {
    std::ifstream file(name, std::ios_base::binary | std::ios_base::in);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file " + name);
    }
    std::vector<char> output {std::istreambuf_iterator<char> {file}, {}};
    return output;
}

// read image data from disk
template <typename DType, typename DeviceMem, typename Backend>
DeviceMem read_image_data(
        std::string const &name, Backend &backend, size_t size) {
    sycl::range<1> r {size}; // resnet input size
    sycl::buffer<DType> b {r};
    auto data = read_binary_data(name);
    assert(data.size() == size * sizeof(DType));
    {
        auto char_data = b.template reinterpret<char>(r * sizeof(DType));
        auto event = backend.get_queue().submit([&](sycl::handler &cgh) {
            auto acc = char_data.template get_access<
                    sycl::access::mode::discard_write>(cgh);
            cgh.copy(data.data(), acc);
        });
        event.wait_and_throw();
    }
    return DeviceMem {b, 0};
}

/*
void dump_desc(const TensorDescriptor &desc) {
    int n, c, h, w;
    int s_n, s_c, s_h, s_w;
    SNNDataType data_type;
    getTensor4dDescriptor(
            desc, &data_type, &n, &c, &h, &w, &s_n, &s_c, &s_h, &s_w);
    std::cout << "\tdims: [" << n << "," << c << "," << h << "," << w << "]\n";
}

void dump_desc(const FilterDescriptor &desc) {
    int k, c, h, w;
    sycldnn::FilterFormat format;
    SNNDataType data_type;
    getFilter4dDescriptor(desc, &data_type, &format, &k, &c, &h, &w);
    std::cout << "\tdims: [" << k << "," << c << "," << h << "," << w << "]\n";
}

void dump_desc(const ConvolutionDescriptor &desc) {
    std::cout << "\tpad: [" << desc.getPadH() << "," << desc.getPadW() << "]\n";
    std::cout << "\tstride: [" << desc.getStrideH() << "," << desc.getStrideW()
              << "]\n";
}

void dump_desc(const PoolingDescriptor &desc) {
    std::cout << "\tmode: " << static_cast<int>(desc.getMode()) << "\n";
    std::cout << "\twindow: [" << desc.getWindowH() << "," << desc.getWindowW()
              << "]\n";
    std::cout << "\tpad: [" << desc.getPadH() << "," << desc.getPadW() << "]\n";
    std::cout << "\tstride: [" << desc.getStrideH() << "," << desc.getStrideW()
              << "]\n";
}
*/
} // namespace helpers

template <typename T>
struct Layer {
    explicit Layer(dnnl::engine &engine, dnnl::stream &stream)
        : engine_(engine), stream_(stream), out_ptr_(nullptr), out_desc_({}) {}

    Layer(dnnl::engine &engine, dnnl::stream &stream,
            const dnnl::memory::desc &out_desc)
        : engine_(engine)
        , stream_(stream)
        , out_ptr_(nullptr)
        , out_desc_(out_desc) {}

    virtual ~Layer() {
        sycl::free(out_ptr_, dnnl::sycl_interop::get_queue(stream_));
    }

    virtual dnnl::status execute(T *in_ptr) = 0;

    T *get_output_ptr() { return out_ptr_; };

    dnnl::memory::desc &get_output_desc() { return out_desc_; }

    /*
    void dump_out_desc() {
        std::cout << "Output:\n";
        helpers::dump_desc(out_desc_);
    }
    */

    virtual void dump_descs() = 0;

    dnnl::engine &engine_;
    dnnl::stream &stream_;

protected:
    T *out_ptr_ = nullptr;
    dnnl::memory::desc out_desc_;
};

template <typename T>
struct ConvBiasLayer : public Layer<T> {
    ConvBiasLayer(dnnl::engine &engine, dnnl::stream &stream,
            std::string const &filter_file, std::string const &bias_file,
            const int in_n, const int in_c, const int in_h, const int in_w,
            const int filt_f, const int filt_c, const int filt_h,
            const int filt_w,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream), eng_(engine), stream_(stream) {
        //auto q = handle.getQueue();

        dnnl::memory::dim N, ic, ih, iw, oc, kh, kw, ph_l, ph_r, pw_l, pw_r, sh,
                sw, oh, ow;

        N = in_n;
        ic = in_c;
        ih = in_h;
        iw = in_w;

        // Read weights from binary file
        auto weights = helpers::read_binary_data(filter_file);
        auto bias_value = helpers::read_binary_data(bias_file);

        kh = filt_h;
        kw = filt_w;
        oc = filt_f;

        //complete formula -> oh= (ih - kh + ph_l + ph_r)/sh + 1;
        oh = ih - kh + 1;
        //complete formula -> ow= (iw - kw + pw_l + pw_r)/sw + 1;
        ow = iw - kw + 1;

        // strides appear to be 1 in portDNN code
        sh = 1;
        sw = 1;

        dnnl::memory::dims src_dims = {in_n, in_c, in_h, in_w};
        dnnl::memory::dims weights_dims = {oc, ic, kh, kw};
        dnnl::memory::dims dst_dims = {in_n, filt_f, oh, ow};
        dnnl::memory::dims bias_dims = {oc};
        dnnl::memory::dims strides_dims = {sh, sw};
        dnnl::memory::dims padding_dims_l = {0, 0};
        dnnl::memory::dims padding_dims_r = {0, 0};
        dnnl::memory::dims dilates = {0, 0};

        const auto sycl_queue = dnnl::sycl_interop::get_queue(stream_);
        this->out_ptr_ = sycl::malloc_device<T>(N * oc * oh * ow, sycl_queue);

        //conv_src_md = dnnl::memory::desc(src_dims, data_type, format);
        conv_src_md = dnnl::memory::desc(
                src_dims, data_type, dnnl::memory::format_tag::nhwc);
        conv_weights_md = dnnl::memory::desc(
                weights_dims, data_type, dnnl::memory::format_tag::iohw);

        this->out_desc_ = dnnl::memory::desc(
                dst_dims, data_type, dnnl::memory::format_tag::nhwc);
        user_bias_md = dnnl::memory::desc(
                bias_dims, data_type, dnnl::memory::format_tag::a);
        //user_bias_md = dnnl::memory::desc();

        //user_src_mem = dnnl::memory(conv_src_md, eng_);
        user_src_mem = dnnl::memory(
                {src_dims, data_type, dnnl::memory::format_tag::nchw}, eng_);
        // I have no idea which filter I should use for this one
        //user_weights_mem = dnnl::memory(conv_weights_md, eng_);
        user_weights_mem = dnnl::memory(
                {weights_dims, data_type, dnnl::memory::format_tag::iohw},
                eng_);

        //user_dst_mem = dnnl::memory(this->out_desc_, eng_);
        user_dst_mem = dnnl::memory(
                {dst_dims, data_type, dnnl::memory::format_tag::nchw}, eng_);

        user_bias_mem = dnnl::memory(user_bias_md, eng_);

        write_to_dnnl_memory(weights.data(), user_weights_mem);
        write_to_dnnl_memory(bias_value.data(), user_bias_mem);

        // Create descriptors for memory above

        conv_pd_ = dnnl::convolution_forward::primitive_desc(eng_,
                dnnl::prop_kind::forward_inference,
                dnnl::algorithm::convolution_auto, conv_src_md, conv_weights_md,
                user_bias_md, this->out_desc_, strides_dims, dilates,
                padding_dims_l, padding_dims_r);
    }

    dnnl::status execute(T *in_ptr) override {
        //std::cout << "zelda " << __LINE__ << ' ' << __FUNCTION__ << '\n';

        write_to_dnnl_memory(in_ptr, user_src_mem);
        // For now, assume that the src, weights, and dst memory layouts generated
        // by the primitive and the ones provided by the user are identical.
        auto conv_src_mem = user_src_mem;
        auto conv_weights_mem = user_weights_mem;
        auto conv_dst_mem = user_dst_mem;

        /*
        // Reorder the data in case the src and weights memory layouts generated by
        // the primitive and the ones provided by the user are different. In this
        // case, we create additional memory objects with internal buffers that will
        // contain the reordered data. The data in dst will be reordered after the
        // convolution computation has finalized.
        if (conv_pd_.src_desc() != user_src_mem.get_desc()) {
            conv_src_mem = dnnl::memory(conv_pd_.src_desc(), eng_);
            dnnl::reorder(user_src_mem, conv_src_mem)
                    .execute(stream_, user_src_mem, conv_src_mem);
        }

        if (conv_pd_.weights_desc() != user_weights_mem.get_desc()) {
            conv_weights_mem = dnnl::memory(conv_pd_.weights_desc(), eng_);
            dnnl::reorder(user_weights_mem, conv_weights_mem)
                    .execute(stream_, user_weights_mem, conv_weights_mem);
        }

        if (conv_pd_.dst_desc() != user_dst_mem.get_desc()) {
            conv_dst_mem = dnnl::memory(conv_pd_.dst_desc(), eng_);
        }
        */

        // Create the primitive.
        auto conv_prim = dnnl::convolution_forward(conv_pd_);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> conv_args;
        /*
        conv_args.insert({DNNL_ARG_SRC, conv_src_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, conv_weights_mem});
        conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
        conv_args.insert({DNNL_ARG_DST, conv_dst_mem});
        */
        conv_args.insert({DNNL_ARG_SRC, user_src_mem});
        conv_args.insert({DNNL_ARG_WEIGHTS, user_weights_mem});
        conv_args.insert({DNNL_ARG_BIAS, user_bias_mem});
        conv_args.insert({DNNL_ARG_DST, user_dst_mem});
        //std::cout << "zelda " << __LINE__ << ' ' << __FUNCTION__ << '\n';

        conv_prim.execute(stream_, conv_args);

        /*
        // Reorder the data in case the dst memory descriptor generated by the
        // primitive and the one provided by the user are different.
        if (conv_pd_.dst_desc() != user_dst_mem.get_desc()) {
            dnnl::reorder(conv_dst_mem, user_dst_mem)
                    .execute(stream_, conv_dst_mem, user_dst_mem);
        } else
            user_dst_mem = conv_dst_mem;
        */

        // Wait for the computation to finalize.
        stream_.wait();

        // Read data from memory object's handle.
        //read_from_dnnl_memory(this->out_ptr_, user_dst_mem);
        read_from_dnnl_memory(this->out_ptr_, user_dst_mem);
        //this->out_desc_ = conv_dst_mem.get_desc();
        //std::cout << "zelda " << __LINE__ << ' ' << __FUNCTION__ << '\n';

        return {};
    }

    /*
    ~ConvBiasLayer() {
        if (ws_ptr_) { sycl::free(ws_ptr_, this->handle_.getQueue()); }
        sycl::free(filt_ptr_, this->handle_.getQueue());
    }
    */

    virtual void dump_descs() override {
        std::cout
                << "I should dump ConvBiasLayer, but still unimplemented Line: "
                << __LINE__ << '\n';
        /*
        std::cout << "Layer type: Convolution\n";
        std::cout << "Conv:\n";
        helpers::dump_desc(conv_desc_);
        std::cout << "Input:\n";
        helpers::dump_desc(in_desc_);
        std::cout << "Filter:\n";
        helpers::dump_desc(filt_desc_);
        */
    }

private:
    size_t ws_size_;
    dnnl::memory user_src_mem;
    dnnl::memory user_weights_mem;
    dnnl::memory user_dst_mem;
    dnnl::memory user_bias_mem;
    dnnl::memory::desc conv_src_md;
    dnnl::memory::desc conv_weights_md;
    dnnl::memory::desc user_bias_md;
    dnnl::memory::desc conv_dst_md;

    const dnnl::engine &eng_;
    dnnl::stream &stream_;
    dnnl::convolution_forward::primitive_desc conv_pd_;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
};

template <typename T>
struct BatchNormLayer : public Layer<T> {
    BatchNormLayer(dnnl::engine &engine, dnnl::stream &stream,
            std::string const &scale_file, std::string const &bias_file,
            std::string const &mean_file, std::string const &var_file,
            const int batch, const int channels, const int rows, const int cols,
            const bool add_relu = true,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream)
        , eng_(engine)
        , stream_(stream)
        , _relu(add_relu) {

        dnnl::memory::dims src_dims = {batch, channels, rows, cols};
        dnnl::memory::dims scaleshift_dims = {channels};
        dnnl::memory::dims mean_dims = {channels};
        dnnl::memory::dims var_dims = {channels};

        auto bias_value = helpers::read_binary_data(bias_file);
        auto scale_value = helpers::read_binary_data(scale_file);
        auto mean_value = helpers::read_binary_data(mean_file);
        auto var_value = helpers::read_binary_data(var_file);

        // Create memory descriptors
        src_md = dnnl::memory::desc(src_dims, data_type, format);
        this->out_desc_ = dnnl::memory::desc(src_dims, data_type, format);
        scaleshift_md = dnnl::memory::desc(
                scaleshift_dims, data_type, dnnl::memory::format_tag::a);

        const auto q = dnnl::sycl_interop::get_queue(stream_);
        this->out_ptr_
                = sycl::malloc_device<T>(batch * channels * rows * cols, q);

        mean_md = dnnl::memory::desc(
                mean_dims, data_type, dnnl::memory::format_tag::x);
        variance_md = dnnl::memory::desc(
                var_dims, data_type, dnnl::memory::format_tag::x);

        src_mem = dnnl::memory(src_md, eng_);
        dst_mem = dnnl::memory(this->out_desc_, eng_);
        scale_mem = dnnl::memory(scaleshift_md, eng_);
        shift_mem = dnnl::memory(scaleshift_md, eng_);

        mean_mem = dnnl::memory(mean_md, eng_);
        variance_mem = dnnl::memory(variance_md, eng_);

        dnnl::normalization_flags flags = (_relu)
                ? (dnnl::normalization_flags::use_scale
                        | dnnl::normalization_flags::use_shift
                        | dnnl::normalization_flags::use_global_stats
                        | dnnl::normalization_flags::fuse_norm_relu)
                : (dnnl::normalization_flags::use_scale
                        | dnnl::normalization_flags::use_shift
                        | dnnl::normalization_flags::use_global_stats);

        bnorm_pd = dnnl::batch_normalization_forward::primitive_desc(eng_,
                dnnl::prop_kind::forward_inference, src_md, this->out_desc_,
                eps_, flags);

        if (_relu)
            workspace_mem = dnnl::memory(bnorm_pd.workspace_desc(), eng_);

        write_to_dnnl_memory(mean_value.data(), mean_mem);
        write_to_dnnl_memory(var_value.data(), variance_mem);
        write_to_dnnl_memory(scale_value.data(), scale_mem);
        write_to_dnnl_memory(bias_value.data(), shift_mem);

    }

    dnnl::status execute(T *in_ptr) override {

        write_to_dnnl_memory(in_ptr, src_mem);

        auto bnorm_prim = dnnl::batch_normalization_forward(bnorm_pd);

        // Primitive arguments. Set up in-place execution by assigning src as DST.
        std::unordered_map<int, dnnl::memory> bnorm_args;
        bnorm_args.insert({DNNL_ARG_SRC, src_mem});
        bnorm_args.insert({DNNL_ARG_MEAN, mean_mem});
        bnorm_args.insert({DNNL_ARG_VARIANCE, variance_mem});
        bnorm_args.insert({DNNL_ARG_SCALE, scale_mem});
        bnorm_args.insert({DNNL_ARG_SHIFT, shift_mem});
        bnorm_args.insert({DNNL_ARG_DST, dst_mem});

        bnorm_prim.execute(stream_, bnorm_args);

        stream_.wait();

        read_from_dnnl_memory(this->out_ptr_, dst_mem);

        return {};
    }

    ~BatchNormLayer() {}

    virtual void dump_descs() override {
        std::cout << "Layer type: BatchNorm dump not implmented yet!\n";
        /*
        std::cout << "Layer type: BatchNorm\n";
        std::cout << "Param:\n";
        helpers::dump_desc(param_desc_);
        std::cout << "In:\n";
        helpers::dump_desc(this->in_desc_);
        */
    }

private:
    //BatchNormMode mode_;
    dnnl::engine &eng_;
    dnnl::stream &stream_;

    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::memory scale_mem;
    dnnl::memory shift_mem;
    dnnl::memory mean_mem;
    dnnl::memory variance_mem;
    dnnl::memory workspace_mem;
    dnnl::memory::desc src_md;
    dnnl::memory::desc scaleshift_md;
    dnnl::memory::desc mean_md;
    dnnl::memory::desc variance_md;
    bool _relu {true};

    dnnl::batch_normalization_forward::primitive_desc bnorm_pd;

    /*
    T *scale_ptr_ = nullptr;
    T *bias_ptr_ = nullptr;
    T *mean_ptr_ = nullptr;
    T *var_ptr_ = nullptr;
    */

    float eps_ = 1.0e-5;
    float alpha_ = 1.0;
    float beta_ = 0.0;
};

template <typename T>
struct GlobalMaxPoolLayer : public Layer<T> {
    GlobalMaxPoolLayer(dnnl::engine &engine, dnnl::stream &stream,
            const int batch, const int channels, const int rows, const int cols,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream), eng_(engine), stream_(stream) {

        dnnl::memory::dim N, ic, ih, iw, kh, kw, padd_l, padd_r, sh, sw, dh, dw,
                oh, ow;

        N = batch;
        ic = channels;
        ih = rows;
        ic = cols;
        kh = rows;
        kw = cols;
        sh = kh;
        sw = kw;
        padd_l = 0;
        padd_r = 0;
        oh = 1;
        ow = 1;

        // I have no idea about what dilation is
        dh = 1;
        dw = 1;

        //dnnl::memory::dims src_dims = {N, ic, ih, iw};
        dnnl::memory::dims src_dims = {batch, channels, rows, cols};

        dnnl::memory::dims dst_dims = {batch, channels, 1, 1};
        dnnl::memory::dims kernel_dims = {kh, kw};
        dnnl::memory::dims strides_dims = {sh, sw};
        dnnl::memory::dims padding_dims_l = {0, 0};
        dnnl::memory::dims padding_dims_r = {0, 0};
        dnnl::memory::dims dilation_dims = {dh, dw};

        src_md = dnnl::memory::desc(src_dims, data_type, format);
        src_mem = dnnl::memory(src_md, eng_);

        this->out_desc_ = dnnl::memory::desc(dst_dims, data_type, format);
        dst_mem = dnnl::memory(this->out_desc_, eng_);

        auto q = dnnl::sycl_interop::get_queue(stream_);
        this->out_ptr_ = sycl::malloc_device<T>(batch * channels, q);

        pooling_pd = dnnl::pooling_forward::primitive_desc(eng_,
                dnnl::prop_kind::forward_training, dnnl::algorithm::pooling_max,
                src_md, this->out_desc_, strides_dims, kernel_dims,
                dilation_dims, padding_dims_l, padding_dims_r);
        // Create workspace memory objects using memory descriptor created by the
        // primitive descriptor.
        // NOTE: Here, the workspace is required to save the indices where maximum
        // was found, and is used in backward pooling to perform upsampling.
        workspace_mem = dnnl::memory(pooling_pd.workspace_desc(), eng_);
    }

    dnnl::status execute(T *in_ptr) override {
        write_to_dnnl_memory(in_ptr, src_mem);
        auto pooling_prim = dnnl::pooling_forward(pooling_pd);
        // Primitive arguments. Set up in-place execution by assigning src as DST.
        std::unordered_map<int, dnnl::memory> pooling_args;
        pooling_args.insert({DNNL_ARG_SRC, src_mem});
        pooling_args.insert({DNNL_ARG_DST, dst_mem});
        pooling_args.insert({DNNL_ARG_WORKSPACE, workspace_mem});

        // Primitive execution: pooling.
        pooling_prim.execute(stream_, pooling_args);

        // Wait for the computation to finalize.
        stream_.wait();

        // Read data from memory object's handle.
        read_from_dnnl_memory(this->out_ptr_, dst_mem);

        return {};
    }
    ~GlobalMaxPoolLayer() = default;

    virtual void dump_descs() override {
        std::cout << "Layer type: GlobalMaxPool\n";
        /*
        std::cout << "Pool:\n";
        helpers::dump_desc(pool_desc_);
        std::cout << "In:\n";
        helpers::dump_desc(in_desc_);
        */
    }

private:
    //TensorDescriptor in_desc_;
    //PoolingDescriptor pool_desc_;
    dnnl::engine &eng_;
    dnnl::stream &stream_;
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::memory workspace_mem;
    dnnl::memory::desc src_md;
    dnnl::pooling_forward::primitive_desc pooling_pd;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
};

template <typename T>
struct FCLayer : public Layer<T> {
    FCLayer(dnnl::engine &engine, dnnl::stream &stream,
            const std::string &weights_file, const std::string &bias_file,
            const int batch, const int in_channels, const int out_channels,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream), eng_(engine), stream_(stream) {
        auto q = dnnl::sycl_interop::get_queue(stream_);

        dnnl::memory::dims src_dims, dst_dims, weights_dims, bias_dims;

        src_dims = {1, batch, in_channels};
        weights_dims = {1, in_channels, out_channels};
        bias_dims = {1, 1, out_channels};
        dst_dims = {1, batch, out_channels};

        src_md = dnnl::memory::desc(
                src_dims, data_type, dnnl::memory::format_tag::abc);
        this->out_desc_ = dnnl::memory::desc(
                dst_dims, data_type, dnnl::memory::format_tag::abc);
        weights_md = dnnl::memory::desc(
                weights_dims, data_type, dnnl::memory::format_tag::abc);
        bias_md = dnnl::memory::desc(
                bias_dims, data_type, dnnl::memory::format_tag::abc);

        auto weights = helpers::read_binary_data(weights_file);
        auto bias = helpers::read_binary_data(bias_file);

        src_mem = dnnl::memory(src_md, eng_);
        weights_mem = dnnl::memory(weights_md, eng_);
        bias_mem = dnnl::memory(bias_md, eng_);
        dst_mem = dnnl::memory(this->out_desc_, eng_);

        write_to_dnnl_memory(weights.data(), weights_mem);
        write_to_dnnl_memory(bias.data(), bias_mem);

        this->out_ptr_ = sycl::malloc_device<T>(batch * out_channels, q);

        matmul_pd = dnnl::matmul::primitive_desc(
                eng_, src_md, weights_md, this->out_desc_);
    }

    dnnl::status execute(T *in_ptr) override {

        write_to_dnnl_memory(in_ptr, src_mem);

        // Create the primitive.
        auto matmul_prim = dnnl::matmul(matmul_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, src_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        matmul_args.insert({DNNL_ARG_BIAS, bias_mem});
        matmul_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: matrix multiplication with ReLU.
        matmul_prim.execute(stream_, matmul_args);

        // Wait for the computation to finalize.
        stream_.wait();

        // Read data from memory object's handle.
        read_from_dnnl_memory(this->out_ptr_, dst_mem);

        return {};
    }

    ~FCLayer() {}

    virtual void dump_descs() override { std::cout << "Layer type: FC\n"; }

private:
    int m_, k_, n_;
    dnnl::engine &eng_;
    dnnl::stream &stream_;
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::memory bias_mem;
    dnnl::memory weights_mem;
    dnnl::memory::desc src_md;
    dnnl::memory::desc weights_md;
    dnnl::memory::desc bias_md;

    dnnl::matmul::primitive_desc matmul_pd;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
};

template <typename T>
struct MMLayer : public Layer<T> {
    MMLayer(dnnl::engine &engine, dnnl::stream &stream, T *lhs_ptr,
            const int batch, const int m, const int k, const int n,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream)
        , eng_(engine)
        , stream_(stream)
        , lhs_ptr_ {lhs_ptr} {

        auto q = dnnl::sycl_interop::get_queue(stream);

        dnnl::memory::dims src_dims = {batch, m, k};
        dnnl::memory::dims weights_dims = {batch, k, n};
        dnnl::memory::dims dst_dims = {batch, m, n};

        this->out_ptr_ = sycl::malloc_device<T>(batch * m * n, q);

        src_desc = dnnl::memory::desc(
                src_dims, data_type, dnnl::memory::format_tag::abc);
        weights_desc = dnnl::memory::desc(
                weights_dims, data_type, dnnl::memory::format_tag::abc);
        this->out_desc_ = dnnl::memory::desc(
                dst_dims, data_type, dnnl::memory::format_tag::abc);

        src_mem = dnnl::memory(src_desc, eng_);
        weights_mem = dnnl::memory(weights_desc, eng_);
        dst_mem = dnnl::memory(this->out_desc_, eng_);

        matmul_pd = dnnl::matmul::primitive_desc(
                eng_, src_desc, weights_desc, this->out_desc_);
    }

    dnnl::status execute(T *rhs_ptr) override {
        write_to_dnnl_memory(rhs_ptr, weights_mem);
        write_to_dnnl_memory(lhs_ptr_, src_mem);
        auto matmul = dnnl::matmul(matmul_pd);

        // Primitive arguments.
        std::unordered_map<int, dnnl::memory> matmul_args;
        matmul_args.insert({DNNL_ARG_SRC, src_mem});
        matmul_args.insert({DNNL_ARG_WEIGHTS, weights_mem});
        matmul_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: matrix multiplication with ReLU.
        matmul.execute(stream_, matmul_args);

        // Wait for the computation to finalize.
        stream_.wait();

        // Read data from memory object's handle.
        read_from_dnnl_memory(this->out_ptr_, dst_mem);

        return {};
    }

    ~MMLayer() = default;

    virtual void dump_descs() override { std::cout << "Layer type: Matmul\n"; }

private:
    T *lhs_ptr_ = nullptr;

    //int batch_, m_, k_, n_;

    dnnl::engine &eng_;
    dnnl::stream &stream_;
    dnnl::memory::desc src_desc;
    dnnl::memory::desc weights_desc;
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::memory weights_mem;
    dnnl::matmul::primitive_desc matmul_pd;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
};

template <typename T>
struct SumLayer : public Layer<T> {
    SumLayer(dnnl::engine &engine, dnnl::stream &stream,
            std::string const &bias_file, const int batch, const int channels,
            const int rows, const int cols,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream), eng_(engine), stream_(stream) {

        auto q = dnnl::sycl_interop::get_queue(stream_);

        dnnl::memory::dims src_dims = {batch, channels, rows, cols};
        dnnl::memory::dims scale_dims = {batch, channels, rows, cols};

        src_desc_ = dnnl::memory::desc(src_dims, data_type, format);
        bias_desc_ = dnnl::memory::desc(scale_dims, data_type, format);
        this->out_desc_ = dnnl::memory::desc(src_dims, data_type, format);

        auto scale_chars = helpers::read_binary_data(bias_file);

        src_mem = dnnl::memory(src_desc_, eng_);
        bias_mem = dnnl::memory(bias_desc_, eng_);
        dst_mem = dnnl::memory(this->out_desc_, eng_);

        this->out_ptr_
                = sycl::malloc_device<T>(batch * channels * rows * cols, q);
        write_to_dnnl_memory(scale_chars.data(), bias_mem);

        sum_pd = dnnl::binary::primitive_desc(eng_, dnnl::algorithm::binary_add,
                src_desc_, bias_desc_, this->out_desc_);
    }

    dnnl::status execute(T *in_ptr) override {
        write_to_dnnl_memory(in_ptr, src_mem);

        auto sum_prim = dnnl::binary(sum_pd);

        // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
        std::unordered_map<int, dnnl::memory> binary_args;
        binary_args.insert({DNNL_ARG_SRC_0, src_mem});
        binary_args.insert({DNNL_ARG_SRC_1, bias_mem});
        binary_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: binary with ReLU.
        sum_prim.execute(stream_, binary_args);

        // Wait for the computation to finalize.
        stream_.wait();

        // Read data from memory object's handle.
        read_from_dnnl_memory(this->out_ptr_, dst_mem);

        return {};
    }
    ~SumLayer() = default;

    virtual void dump_descs() override {
        std::cout << "Layer type: Sum\n";
        /*
        std::cout << "Bias:\n";
        helpers::dump_desc(bias_desc_);
        std::cout << "In:\n";
        helpers::dump_desc(this->out_desc_);
        */
    }

private:
    dnnl::engine &eng_;
    dnnl::stream &stream_;
    dnnl::memory::desc src_desc_;
    dnnl::memory::desc bias_desc_;
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::memory bias_mem;
    dnnl::binary::primitive_desc sum_pd;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
};

template <typename T>
struct SoftmaxLayer : public Layer<T> {
    SoftmaxLayer(dnnl::engine &engine, dnnl::stream &stream, const int batch,
            const int channels, const int rows, const int cols,
            dnnl::algorithm algo = dnnl::algorithm::softmax_accurate,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream), eng_(engine), stream_(stream) {

        auto q = dnnl::sycl_interop::get_queue(stream_);

        dnnl::memory::dims src_dst_dims = {batch, channels, rows, cols};
        src_md = dnnl::memory::desc(src_dst_dims, data_type, format);
        this->out_desc_ = dnnl::memory::desc(src_dst_dims, data_type, format);

        this->out_ptr_
                = sycl::malloc_device<T>(batch * channels * rows * cols, q);

        src_mem = dnnl::memory(src_md, eng_);
        dst_mem = dnnl::memory(this->out_desc_, eng_);
        constexpr int axis = 1;

        softmax_pd = dnnl::softmax_forward::primitive_desc(eng_,
                dnnl::prop_kind::forward_training, algo, src_md,
                this->out_desc_, axis);
    }

    dnnl::status execute(T *in_ptr) override {
        write_to_dnnl_memory(in_ptr, src_mem);
        //
        // Create the primitive.
        auto softmax_prim = dnnl::softmax_forward(softmax_pd);

        // Primitive arguments. Set up in-place execution by assigning src as DST.
        std::unordered_map<int, dnnl::memory> softmax_args;
        softmax_args.insert({DNNL_ARG_SRC, src_mem});
        softmax_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution.
        softmax_prim.execute(stream_, softmax_args);

        // Wait for the computation to finalize.
        stream_.wait();

        // Read data from memory object's handle.
        read_from_dnnl_memory(this->out_ptr_, dst_mem);
        return {};
    }
    ~SoftmaxLayer() = default;

    virtual void dump_descs() override {
        std::cout << "Layer type: Softmax\n";
        /*
        std::cout << "In:\n";
        helpers::dump_desc(in_desc_);
        */
    }

private:
    dnnl::engine &eng_;
    dnnl::stream &stream_;

    dnnl::memory::desc src_md;
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::softmax_forward::primitive_desc softmax_pd;

    float alpha_ = 1.0f;
    float beta_ = 0.0f;
};

template <typename T>
struct LogLayer : public Layer<T> {
    LogLayer(dnnl::engine &engine, dnnl::stream &stream, const int batch,
            const int channels, const int rows, const int cols,
            dnnl::memory::format_tag format = dnnl::memory::format_tag::nhwc,
            dnnl::memory::data_type data_type = dnnl::memory::data_type::f32)
        : Layer<T>(engine, stream), eng_(engine), stream_(stream) {

        auto q = dnnl::sycl_interop::get_queue(stream_);

        src_md = dnnl::memory::desc(
                {batch, channels, rows, cols}, data_type, format);
        src_mem = dnnl::memory(src_md, eng_);

        this->out_ptr_
                = sycl::malloc_device<T>(batch * channels * rows * cols, q);
        this->out_desc_ = dnnl::memory::desc(
                {batch, channels, rows, cols}, data_type, format);
        dst_mem = dnnl::memory(this->out_desc_, eng_);

        eltwise_log_pd = dnnl::eltwise_forward::primitive_desc(eng_,
                dnnl::prop_kind::forward_training, dnnl::algorithm::eltwise_log,
                src_md, this->out_desc_);
    }

    dnnl::status execute(T *in_ptr) override {
        write_to_dnnl_memory(in_ptr, src_mem);

        auto eltwise_log = dnnl::eltwise_forward(eltwise_log_pd);

        std::unordered_map<int, dnnl::memory> eltwise_args;
        eltwise_args.insert({DNNL_ARG_SRC, src_mem});
        eltwise_args.insert({DNNL_ARG_DST, dst_mem});

        // Primitive execution: element-wise (ReLU).
        eltwise_log.execute(stream_, eltwise_args);

        // Wait for the computation to finalize.
        stream_.wait();

        // Read data from memory object's handle.
        read_from_dnnl_memory(this->out_ptr_, dst_mem);

        return {};
    }
    ~LogLayer() = default;

    virtual void dump_descs() override {
        std::cout << "Layer type: Activation\n";
        /*
        std::cout << "In:\n";
        helpers::dump_desc(in_desc_);
        */
    }

private:
    dnnl::engine &eng_;
    dnnl::stream &stream_;
    dnnl::memory::desc src_md;
    dnnl::memory src_mem;
    dnnl::memory dst_mem;
    dnnl::eltwise_forward::primitive_desc eltwise_log_pd;
};

template <typename T>
struct Network {
    void add_layer(std::unique_ptr<Layer<T>> layer) {
        layers.emplace_back(std::move(layer));
    }

    void execute(T *in_ptr) {
        for (auto &layer : layers) {
            layer->execute(in_ptr);
#if DEBUG_LAYERS
            layer->handle_.getQueue().wait_and_throw();
            layer->dump_descs();
            layer->dump_out_desc();
            std::cout << "***\n";
#endif
            in_ptr = layer->get_output_ptr();
        }
    }

    dnnl::memory::desc &get_last_output_desc() {
        return layers.back()->get_output_desc();
    }

    T *get_last_output_ptr() {
        return layers.back()->get_output_ptr();
    }

    std::vector<T> get_output_as_host_vec() {
        auto &last_layer = layers.back();
        auto &output_desc = last_layer->get_output_desc();
        auto &stream = last_layer->stream_;
        auto q = dnnl::sycl_interop::get_queue(stream);
        auto tmp = output_desc.get_dims();
        int output_dim {1};
        //std::cout << "output tmp dims: \n";
        for (const auto &e : tmp) {
            output_dim *= e;
        }
        //std::cout << "\n full dim: " << output_dim << '\n';
        //std::cout << std::endl;
        std::vector<T> output(output_dim);
        q.wait();

        q.memcpy(output.data(), last_layer->get_output_ptr(),
                 output_dim * sizeof(T))
                .wait_and_throw();

        return output;
    }

    void dump_output() {
        auto output = get_output_as_host_vec();
        std::cout << "Output:\n";
        for (auto e : output) {
            std::cout << e << ", ";
        }
        std::cout << "\n";
    }

    std::vector<std::unique_ptr<Layer<T>>> layers;
};

template <typename T>
inline void add_conv_bias_layer(Network<T> &net, dnnl::engine &handle,
        dnnl::stream &stream, std::string const &filter_file,
        std::string const &bias_file, const int in_n, const int in_c,
        const int in_h, const int in_w, const int filt_f, const int filt_c,
        const int filt_h, const int filt_w) {
    net.add_layer(std::make_unique<ConvBiasLayer<T>>(handle, stream,
            filter_file, bias_file, in_n, in_c, in_h, in_w, filt_f, filt_c,
            filt_h, filt_w));
}

template <typename T>
inline void add_batchnorm_layer(Network<T> &net, dnnl::engine &handle,
        dnnl::stream &stream, std::string const &scale_file,
        std::string const &bias_file, std::string const &mean_file,
        std::string const &var_file, const int n, const int c, const int h,
        const int w, const bool add_relu = true) {
    net.add_layer(std::make_unique<BatchNormLayer<T>>(handle, stream,
            scale_file, bias_file, mean_file, var_file, n, c, h, w, add_relu));
}

template <typename T>
inline void add_global_max_pool_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, const int n, const int c, const int h,
        const int w) {
    net.add_layer(std::make_unique<GlobalMaxPoolLayer<T>>(
            engine, stream, n, c, h, w));
}

template <typename T>
inline void add_fc_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, const std::string &weights_file,
        const std::string &bias_file, const int batch, const int in_c,
        const int out_c) {
    net.add_layer(std::make_unique<FCLayer<T>>(
            engine, stream, weights_file, bias_file, batch, in_c, out_c));
}

template <typename T>
inline void add_mm_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, T *lhs_ptr, const int batch, const int m,
        const int k, const int n) {
    net.add_layer(std::make_unique<MMLayer<T>>(
            engine, stream, lhs_ptr, batch, m, k, n));
}

template <typename T>
inline void add_softmax_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, const int n, const int c, const int h,
        const int w) {
    net.add_layer(
            std::make_unique<SoftmaxLayer<T>>(engine, stream, n, c, h, w));
}

template <typename T>
inline void add_log_layer(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, const int n, const int c, const int h,
        const int w) {
    net.add_layer(std::make_unique<LogLayer<T>>(engine, stream, n, c, h, w));
}

template <typename T>
inline void add_sum_layer(Network<T> &net, dnnl::engine &handle,
        dnnl::stream &stream, std::string const &bias_file, const int n,
        const int c, const int h, const int w) {
    net.add_layer(std::make_unique<SumLayer<T>>(
            handle, stream, bias_file, n, c, h, w));
}

template <typename T>
inline void add_conv_bias_bnorm_relu_block(Network<T> &net,
        dnnl::engine &engine, dnnl::stream &stream,
        std::string const &file_directory, std::string const &conv_filter_file,
        std::string const &conv_bias_file, std::string const &bn_scale_file,
        std::string const &bn_bias_file, std::string const &bn_mean_file,
        std::string const &bn_var_file, const int in_n, const int in_c,
        const int in_h, const int in_w, const int out_c, const int filt_h,
        const int filt_w, bool add_relu = true) {
    add_conv_bias_layer(net, engine, stream, file_directory + conv_filter_file,
            file_directory + conv_bias_file, in_n, in_c, in_h, in_w, out_c,
            in_c, filt_h, filt_w);
    add_batchnorm_layer(net, engine, stream, file_directory + bn_scale_file,
            file_directory + bn_bias_file, file_directory + bn_mean_file,
            file_directory + bn_var_file, in_n, out_c, in_h, in_w, add_relu);
}

template <typename T>
inline void add_fc_bias_bnorm_relu_block(Network<T> &net, dnnl::engine &engine,
        dnnl::stream &stream, std::string const &file_directory,
        std::string const &fc_filter_file, std::string const &fc_bias_file,
        std::string const &bn_scale_file, std::string const &bn_bias_file,
        std::string const &bn_mean_file, std::string const &bn_var_file,
        const int batch, const int in_c, const int out_c) {
    add_fc_layer(net, engine, stream, file_directory + fc_filter_file,
            file_directory + fc_bias_file, batch, in_c, out_c);
    add_batchnorm_layer(net, engine, stream, file_directory + bn_scale_file,
            file_directory + bn_bias_file, file_directory + bn_mean_file,
            file_directory + bn_var_file, batch, out_c, 1, 1);
    /*
    add_activation_layer(
            net, handle, batch, out_c, 1, 1, ActivationMode::ACTIVATION_RELU);
            */
}

void pointnet(dnnl::engine::kind engine_kind) {}

int main(int argc, char *argv[]) {

    if (argc < 3) {
        std::cout << "USAGE: " << argv[0] << " <directory> <image>"
                  << std::endl;
        return 1;
    }

    dnnl::engine eng(dnnl::engine::kind::gpu, 0);
    dnnl::stream stream(eng);
    auto sycl_queue = dnnl::sycl_interop::get_queue(stream);

    std::string data_dir {argv[1]};
    data_dir += "/";

    DType *in_ptr = sycl::malloc_device<DType>(32 * 1024 * 3, sycl_queue);
    auto input = helpers::read_binary_data(argv[2]);
    helpers::copy_to_device(input, in_ptr, sycl_queue);

    Network<DType> input_transform_block;
    Network<DType> base_transform_block;
    Network<DType> feature_transform_block;

    // Construct input transformation block of network
    add_conv_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.conv1.weight.bin",
            "transform.input_transform.conv1.bias.bin",
            "transform.input_transform.bn1.weight.bin",
            "transform.input_transform.bn1.bias.bin",
            "transform.input_transform.bn1.running_mean.bin",
            "transform.input_transform.bn1.running_var.bin", 32, 3, 1024, 1, 64,
            1, 1);

    add_conv_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.conv2.weight.bin",
            "transform.input_transform.conv2.bias.bin",
            "transform.input_transform.bn2.weight.bin",
            "transform.input_transform.bn2.bias.bin",
            "transform.input_transform.bn2.running_mean.bin",
            "transform.input_transform.bn2.running_var.bin", 32, 64, 1024, 1,
            128, 1, 1);

    add_conv_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.conv3.weight.bin",
            "transform.input_transform.conv3.bias.bin",
            "transform.input_transform.bn3.weight.bin",
            "transform.input_transform.bn3.bias.bin",
            "transform.input_transform.bn3.running_mean.bin",
            "transform.input_transform.bn3.running_var.bin", 32, 128, 1024, 1,
            1024, 1, 1);

    add_global_max_pool_layer(
            input_transform_block, eng, stream, 32, 1024, 1024, 1);

    add_fc_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.fc1.weight.bin",
            "transform.input_transform.fc1.bias.bin",
            "transform.input_transform.bn4.weight.bin",
            "transform.input_transform.bn4.bias.bin",
            "transform.input_transform.bn4.running_mean.bin",
            "transform.input_transform.bn4.running_var.bin", 32, 1024, 512);

    add_fc_bias_bnorm_relu_block(input_transform_block, eng, stream, data_dir,
            "transform.input_transform.fc2.weight.bin",
            "transform.input_transform.fc2.bias.bin",
            "transform.input_transform.bn5.weight.bin",
            "transform.input_transform.bn5.bias.bin",
            "transform.input_transform.bn5.running_mean.bin",
            "transform.input_transform.bn5.running_var.bin", 32, 512, 256);

    add_fc_layer(input_transform_block, eng, stream,
            data_dir + "transform.input_transform.fc3.weight.bin",
            data_dir + "transform.input_transform.fc3.bias.bin", 32, 256, 9);

    // TODO : convert bias layer to sum layers requires more changes than expected,
    // going back to it later
    add_sum_layer(input_transform_block, eng, stream,
            data_dir + "transform.input_transform.id.bin", 1, 32 * 9, 1, 1);

    // Transform input
    add_mm_layer(input_transform_block, eng, stream, in_ptr, 32, 1024, 3, 3);

    // Construct base transformation block
    add_conv_bias_bnorm_relu_block(base_transform_block, eng, stream, data_dir,
            "transform.conv1.weight.bin", "transform.conv1.bias.bin",
            "transform.bn1.weight.bin", "transform.bn1.bias.bin",
            "transform.bn1.running_mean.bin", "transform.bn1.running_var.bin",
            32, 3, 1024, 1, 64, 1, 1);

    // Construct feature transformation block
    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.feature_transform.conv1.weight.bin",
            "transform.feature_transform.conv1.bias.bin",
            "transform.feature_transform.bn1.weight.bin",
            "transform.feature_transform.bn1.bias.bin",
            "transform.feature_transform.bn1.running_mean.bin",
            "transform.feature_transform.bn1.running_var.bin", 32, 64, 1024, 1,
            64, 1, 1);
    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.feature_transform.conv2.weight.bin",
            "transform.feature_transform.conv2.bias.bin",
            "transform.feature_transform.bn2.weight.bin",
            "transform.feature_transform.bn2.bias.bin",
            "transform.feature_transform.bn2.running_mean.bin",
            "transform.feature_transform.bn2.running_var.bin", 32, 64, 1024, 1,
            128, 1, 1);

    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.feature_transform.conv3.weight.bin",
            "transform.feature_transform.conv3.bias.bin",
            "transform.feature_transform.bn3.weight.bin",
            "transform.feature_transform.bn3.bias.bin",
            "transform.feature_transform.bn3.running_mean.bin",
            "transform.feature_transform.bn3.running_var.bin", 32, 128, 1024, 1,
            1024, 1, 1);

    add_global_max_pool_layer(
            feature_transform_block, eng, stream, 32, 1024, 1024, 1);

    add_fc_bias_bnorm_relu_block(feature_transform_block, eng, stream, data_dir,
            "transform.feature_transform.fc1.weight.bin",
            "transform.feature_transform.fc1.bias.bin",
            "transform.feature_transform.bn4.weight.bin",
            "transform.feature_transform.bn4.bias.bin",
            "transform.feature_transform.bn4.running_mean.bin",
            "transform.feature_transform.bn4.running_var.bin", 32, 1024, 512);

    add_fc_bias_bnorm_relu_block(feature_transform_block, eng, stream, data_dir,
            "transform.feature_transform.fc2.weight.bin",
            "transform.feature_transform.fc2.bias.bin",
            "transform.feature_transform.bn5.weight.bin",
            "transform.feature_transform.bn5.bias.bin",
            "transform.feature_transform.bn5.running_mean.bin",
            "transform.feature_transform.bn5.running_var.bin", 32, 512, 256);

    add_fc_layer(feature_transform_block, eng, stream,
            data_dir + "transform.feature_transform.fc3.weight.bin",
            data_dir + "transform.feature_transform.fc3.bias.bin", 32, 256,
            4096);

    // TODO:: convert to sum
    add_sum_layer(feature_transform_block, eng, stream,
            data_dir + "transform.feature_transform.id.bin", 1, 32 * 4096, 1,
            1);
    add_mm_layer(feature_transform_block, eng, stream,
            base_transform_block.get_last_output_ptr(), 32, 1024, 64, 64);

    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.conv2.weight.bin", "transform.conv2.bias.bin",
            "transform.bn2.weight.bin", "transform.bn2.bias.bin",
            "transform.bn2.running_mean.bin", "transform.bn2.running_var.bin",
            32, 64, 1024, 1, 128, 1, 1);

    add_conv_bias_bnorm_relu_block(feature_transform_block, eng, stream,
            data_dir, "transform.conv3.weight.bin", "transform.conv3.bias.bin",
            "transform.bn3.weight.bin", "transform.bn3.bias.bin",
            "transform.bn3.running_mean.bin", "transform.bn3.running_var.bin",
            32, 128, 1024, 1, 1024, 1, 1, false);

    add_global_max_pool_layer(
            feature_transform_block, eng, stream, 32, 1024, 1, 1024);

    add_fc_bias_bnorm_relu_block(feature_transform_block, eng, stream, data_dir,
            "fc1.weight.bin", "fc1.bias.bin", "bn1.weight.bin", "bn1.bias.bin",
            "bn1.running_mean.bin", "bn1.running_var.bin", 32, 1024, 512);

    add_fc_bias_bnorm_relu_block(feature_transform_block, eng, stream, data_dir,
            "fc2.weight.bin", "fc2.bias.bin", "bn2.weight.bin", "bn2.bias.bin",
            "bn2.running_mean.bin", "bn2.running_var.bin", 32, 512, 256);

    add_fc_layer(feature_transform_block, eng, stream,
            data_dir + "fc3.weight.bin", data_dir + "fc3.bias.bin", 32, 256,
            10);

    add_softmax_layer(feature_transform_block, eng, stream, 32, 10, 1, 1);

    add_log_layer(feature_transform_block, eng, stream, 32, 10, 1, 1);

    input_transform_block.execute(in_ptr);
    base_transform_block.execute(input_transform_block.get_last_output_ptr());
    feature_transform_block.execute(base_transform_block.get_last_output_ptr());

    auto output = feature_transform_block.get_output_as_host_vec();

    /*
    for (int i = 0; i < 32 * 10; ++i) {
        std::cout << output[i] << ' ';
    }
    std::cout << '\n';
    */

    // Find index of max value in each row of output, then calculate mode of
    // results to get final classification
    std::vector<int> predicted(32);
    for (int i = 0; i < 32; i++) {
        auto maxVal = std::max_element(
                output.begin() + (i * 10), output.begin() + (i * 10) + 10);
        predicted[i] = std::distance(output.begin() + (i * 10), maxVal);
        //std::cout << "maxVal " << *maxVal << " predicted " << predicted[i]
        //<< " for i=" << i << '\n';
    }
    std::sort(predicted.begin(), predicted.end());

    /*
    for (const auto &e : predicted) {
        std::cout << e << ' ';
    }
    std::cout << '\n';
    */

    int prev = predicted[0];
    int count = 1;
    int mode = 0;
    int mode_count = 0;
    for (size_t i = 1; i < predicted.size(); ++i) {
        if (predicted[i] == prev) {
            count++;
        } else {
            if (count > mode_count) {
                mode = prev;
                mode_count = count;
            }
            count = 1;
        }
        prev = predicted[i];
    }
    if (count > mode_count) {
        mode = prev;
        mode_count = count;
    }
    std::cout << "classed as " << mode << " (i.e., " << object_map[mode] << ")"
              << std::endl;

    /*
    int loops = 8;
    do {
        auto start = std::chrono::high_resolution_clock::now();
        input_transform_block.execute(in_ptr);
        base_transform_block.execute(
                input_transform_block.get_last_output_ptr());
        feature_transform_block.execute(
                base_transform_block.get_last_output_ptr());
        sycl_queue.wait_and_throw();
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << (end - start).count() << " ns" << std::endl;
    } while (--loops);
    */

    sycl::free(in_ptr, sycl_queue);

    return 0;
}
