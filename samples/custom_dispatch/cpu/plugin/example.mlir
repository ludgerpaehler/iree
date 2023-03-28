// RUN: [[ $IREE_TEST_STANDALONE_PLUGIN == 1 ]] || \
// RUN: iree-compile %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/plugin/standalone_plugin.sos \
// RUN:     --function=mixed_invocation \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 | \
// RUN: FileCheck %s --check-prefix=CHECK-STANDALONE

// CHECK-STANDALONE: EXEC @mixed_invocation
// CHECK-STANDALONE: 8xf32=12 12 12 12 12 12 12 12

// RUN: [[ $IREE_TEST_SYSTEM_PLUGIN == 1 ]] || \
// RUN: iree-compile %s | \
// RUN: iree-run-module \
// RUN:     --device=local-sync \
// RUN:     --executable_plugin=$IREE_BINARY_DIR/samples/custom_dispatch/cpu/plugin/system_plugin$IREE_DYLIB_EXT \
// RUN:     --function=mixed_invocation \
// RUN:     --input=8xf32=2 \
// RUN:     --input=8xf32=4 | \
// RUN: FileCheck %s --check-prefix=CHECK-SYSTEM

// CHECK-SYSTEM: EXEC @mixed_invocation
// simple_mul_workgroup
// CHECK-SYSTEM: mul[0](2 * 4 = 8)
// CHECK-SYSTEM: mul[1](2 * 4 = 8)
// CHECK-SYSTEM: mul[2](2 * 4 = 8)
// CHECK-SYSTEM: mul[3](2 * 4 = 8)
// CHECK-SYSTEM: mul[4](2 * 4 = 8)
// CHECK-SYSTEM: mul[5](2 * 4 = 8)
// CHECK-SYSTEM: mul[6](2 * 4 = 8)
// CHECK-SYSTEM: mul[7](2 * 4 = 8)
// arith.addf 8 + 4 = 12
// CHECK-SYSTEM: 8xf32=12 12 12 12 12 12 12 12

// This example demonstrates calling dynamically imported functions in the
// runtime. Alternatively the functions can be embedded into the compiled IREE
// programs for hermetic deployment (see custom_dispatch/cpu/embedded/).

// The configurations used for executable compilation.
// This lets the compiler and runtime know the format and requirements of the
// executable binaries produced. For this sample this matches what we expect
// the plugins to be compiled for (architecture-wise).
#aarch64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-arm_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 16 : index,
  target_triple = "aarch64-unknown-unknown-eabi-elf"
}>
#x86_64_target = #hal.executable.target<"llvm-cpu", "embedded-elf-x86_64", {
  data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128",
  native_vector_size = 32 : index,
  target_triple = "x86_64-unknown-unknown-eabi-elf"
}>

// The target devices that the program will run on.
// These can come from compiler flags and multiple targets can be supported
// It's possible, for example, to support targeting multiple devices in the same
// compiled binary (CPU + Vulkan, etc).
#cpu_target = #hal.device.target<"llvm-cpu", {
  executable_targets = [
    #aarch64_target,
    #x86_64_target
  ]
}>

module @example attributes {hal.device.targets = [#cpu_target]} {

  // Executable containing exported shims and calls to external functions.
  // Each executable can contain multiple exported functions and variants for
  // different architectures or even devices. It's also possible to mix hand-
  // authored functions with code generated ones even for the same functions
  // such that code generation is used as a fallback when the hand-authored
  // kernels aren't supported at runtime.
  stream.executable private @executable {
    stream.executable.export public @simple_mul workgroups(%workload: index) -> (index, index, index) {
      // This host function is used to compute the XYZ workgroup count
      // dispatched at runtime. It can query the %device for capabilities
      // and limits (last-level cache sizes, etc). The other arguments are the
      // values passed in the dispatch operation (usually things like root
      // output op tensor dimensions and other abstract values).
      %x = affine.apply affine_map<()[s0] -> (s0 ceildiv 64)>()[%workload]
      %c1 = arith.constant 1 : index
      stream.return %x, %c1, %c1 : index, index, index
    }

    builtin.module {
      // External function declaration using a user-chosen calling convention.
      func.func private @simple_mul_workgroup(%binding0: memref<?xf32>, %binding1: memref<?xf32>, %binding2: memref<?xf32>, %dim: index, %tid: index)

      // IREE exported function using stream bindings and operands.
      // Compiler passes will be able to optimize across this interface and
      // deduplicate bindings/operands, convert/pack operands, and inline
      // constants operands.
      func.func @simple_mul(
          %binding0: !stream.binding,
          %binding1: !stream.binding,
          %binding2: !stream.binding,
          %dim: index) {
        %c0 = arith.constant 0 : index

        // This function is invoked once per workgroup so determine where this
        // particular workgroup is in the grid. In this example we use a
        // workgroup size of 64x1x1 (which is exceedingly small for CPUs but
        // useful for demonstration).
        %workgroup_id_x = flow.dispatch.workgroup.id[0] : index
        %tid = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%workgroup_id_x]

        // Bindings are accessed by reference.
        %memref0 = stream.binding.subspan %binding0[%c0] : !stream.binding -> memref<?xf32>{%dim}
        %memref1 = stream.binding.subspan %binding1[%c0] : !stream.binding -> memref<?xf32>{%dim}
        %memref2 = stream.binding.subspan %binding2[%c0] : !stream.binding -> memref<?xf32>{%dim}

        // Call the externally defined C function with an (almost) plain C
        // calling convention (see above for details about the mess memrefs
        // turn into). This will be fetched at runtime from the plugin binary.
        func.call @simple_mul_workgroup(%memref0, %memref1, %memref2, %dim, %tid) : (memref<?xf32>, memref<?xf32>, memref<?xf32>, index, index) -> ()

        // NOTE: this is code generated as normal - other MLIR ops can be used
        // here for looping/control flow, vector operations, linalg, etc.
        // This simple sample is just calling out to the external function but
        // microkernels fused with other code are possible.

        return
      }
    }
  }

  // Function demonstrating executable plugins and mixing plugins and codegen.
  // Invoke with:
  //  --device=local-sync
  //  --executable_plugin=standalone_plugin.sos
  //  --function=mixed_invocation
  //  --input=8xf32=2
  //  --input=8xf32=4
  func.func @mixed_invocation(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
    // The only externally available metadata in the dispatch are the values
    // passed in as operands. Here we pass in the dynamic dimension.
    %c0 = arith.constant 0 : index
    %dim = tensor.dim %arg0, %c0 : tensor<?xf32>

    // Dispatch a basic `ret = lhs * rhs` using an external function.
    // This form (@executable::@export) allows for automatic variant selection
    // when multi-targeting.
    %0 = flow.dispatch @executable::@simple_mul[%dim](%arg0, %arg1, %dim) : (tensor<?xf32>{%dim}, tensor<?xf32>{%dim}, index) -> tensor<?xf32>{%dim}

    // Code gen some other ops - these will interleave with hand-authored
    // ones but naturally won't be able to fuse with them.
    %1 = arith.addf %0, %arg1 : tensor<?xf32>

    return %1 : tensor<?xf32>
  }

}  // module
