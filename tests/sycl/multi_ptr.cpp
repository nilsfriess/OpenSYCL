/*
 * This file is part of hipSYCL, a SYCL implementation based on CUDA/HIP
 *
 * Copyright (c) 2024 Aksel Alpay and contributors
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "sycl_test_suite.hpp"

using namespace cl;

BOOST_AUTO_TEST_SUITE(multi_ptr_test_suite)

using sycl::access::address_space;
using sycl::access::decorated;

BOOST_AUTO_TEST_CASE(multi_ptr_api) {
  // Construction
  sycl::multi_ptr<int, address_space::global_space, decorated::no> ptr1;
  sycl::multi_ptr<int, address_space::global_space, decorated::no> ptr2{ptr1};
  sycl::multi_ptr<int, address_space::global_space, decorated::no> ptr3{std::move(ptr2)};
  sycl::multi_ptr<int, address_space::global_space, decorated::yes> ptr4{ptr3};
  sycl::multi_ptr<int, address_space::global_space, decorated::no> ptr5{nullptr};

  // Assignment
  ptr5 = ptr1;
  ptr5 = std::move(ptr3);
  ptr4 = nullptr;

  // Conversion from generic_space pointer -> {global, local, private}_space pointer
  sycl::multi_ptr<int, address_space::generic_space, decorated::no> ptr6;
  sycl::multi_ptr<int, address_space::global_space, decorated::no> ptr7{ptr6};
  sycl::multi_ptr<int, address_space::local_space, decorated::no> ptr8{ptr6};
  sycl::multi_ptr<int, address_space::private_space, decorated::no> ptr9{ptr6};

  // access operators
  int a = 1;
  sycl::multi_ptr<int, address_space::generic_space, decorated::no> ptr10{&a};

  ptr10[0] = 2;
  BOOST_CHECK(a == 2);

  *ptr10 = 3;
  BOOST_CHECK(a == 3);

  int *b = ptr10.get();
  *b = 5;
  BOOST_CHECK(a == 5);

  int *c = ptr10; // Implicit conversion to underlying pointer type
  *c = 6;
  BOOST_CHECK(a == 6);
}

BOOST_AUTO_TEST_CASE(multi_ptr_void) {
  {
    sycl::multi_ptr<int, address_space::global_space, decorated::no> ptr1;
    sycl::multi_ptr<void, address_space::global_space, decorated::no> ptr2 = ptr1;
  }

  {
    sycl::multi_ptr<const int, address_space::global_space, decorated::no> ptr1;
    sycl::multi_ptr<const void, address_space::global_space, decorated::no> ptr2 = ptr1;
  }
}

template <decorated IsDecorated> void multi_ptr_accessor_test() {
  constexpr auto size = 10;

  std::array<int, size> data1;
  std::array<int, size> data2;
  std::array<int, size> data3;

  for (int i = 0; i < size; ++i) {
    data1[i] = 1;
    data2[i] = 2;
    data3[i] = 3;
  }

  {
    sycl::buffer buf1{data1};
    sycl::buffer buf2{data2};
    sycl::buffer buf3{data3};

    sycl::queue{}.submit([&](sycl::handler &cgh) {
      auto acc1 = buf1.get_access(cgh);
      auto acc2 = buf2.get_access(cgh);
      auto acc3 = buf3.get_access(cgh);

      cgh.parallel_for(size, [=](auto i) {
        auto ptr1 = acc1.get_multi_ptr<IsDecorated>();
        sycl::multi_ptr ptr2(acc2);
        // clang-format off
	sycl::multi_ptr<int, address_space::global_space, IsDecorated> ptr3(acc3);
        // clang-format on

        ptr1[i] *= 2;
        ptr2[i] *= 2;
        *(ptr3 + i) *= 2; // test dereferencing
      });
    });
  }

  for (int i = 0; i < size; ++i) {
    BOOST_CHECK(data1[i] == 2 * 1);
    BOOST_CHECK(data2[i] == 2 * 2);
    BOOST_CHECK(data3[i] == 2 * 3);
  }
}

BOOST_AUTO_TEST_CASE(multi_ptr_accessor) {
  multi_ptr_accessor_test<decorated::no>();
  multi_ptr_accessor_test<decorated::yes>();
  multi_ptr_accessor_test<decorated::legacy>();
}

struct S {
  int a;
};

BOOST_AUTO_TEST_CASE(multi_ptr_arrow_op) {
  S s[1] = {1};

  {
    sycl::buffer<S, 1> buf{s, 1};

    sycl::queue{}.submit([&](sycl::handler &cgh) {
      auto acc1 = buf.get_access(cgh);

      cgh.parallel_for(1, [=](auto i) {
        auto ptr = acc1.get_multi_ptr<decorated::no>();
        ptr->a = 2;
      });
    });
  }

  BOOST_CHECK(s[0].a == 2);
}

BOOST_AUTO_TEST_SUITE_END()
