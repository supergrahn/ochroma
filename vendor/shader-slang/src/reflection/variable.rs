use super::{Generic, Type, UserAttribute, rcall};
use crate::{GlobalSession, Modifier, ModifierID, succeeded, sys};

#[repr(transparent)]
pub struct Variable(sys::SlangReflectionVariable);

impl Variable {
	pub fn name(&self) -> &str {
		let name = rcall!(spReflectionVariable_GetName(self));
		unsafe { std::ffi::CStr::from_ptr(name).to_str().unwrap() }
	}

	pub fn ty(&self) -> &Type {
		rcall!(spReflectionVariable_GetType(self) as &Type)
	}

	pub fn find_modifier(&self, id: ModifierID) -> Option<&Modifier> {
		rcall!(spReflectionVariable_FindModifier(self, id) as Option<&Modifier>)
	}

	pub fn user_attribute_count(&self) -> u32 {
		rcall!(spReflectionVariable_GetUserAttributeCount(self))
	}

	pub fn user_attribute_by_index(&self, index: u32) -> Option<&UserAttribute> {
		rcall!(spReflectionVariable_GetUserAttribute(self, index) as Option<&UserAttribute>)
	}

	pub fn user_attributes(&self) -> impl ExactSizeIterator<Item = &UserAttribute> {
		(0..self.user_attribute_count())
			.map(move |i| rcall!(spReflectionVariable_GetUserAttribute(self, i) as &UserAttribute))
	}

	pub fn find_user_attribute_by_name(
		&self,
		global_session: &GlobalSession,
		name: &str,
	) -> Option<&UserAttribute> {
		let name = std::ffi::CString::new(name).unwrap();
		rcall!(spReflectionVariable_FindUserAttributeByName(
			self,
			global_session as *const _ as *mut _,
			name.as_ptr()
		) as Option<&UserAttribute>)
	}

	pub fn has_default_value(&self) -> bool {
		rcall!(spReflectionVariable_HasDefaultValue(self))
	}

	pub fn default_value_int(&self) -> Option<i64> {
		// PATCHED for Track S (Ochroma spectra-native): the FFI symbol
		// `spReflectionVariable_GetDefaultValueInt` does not exist in the Slang
		// 2024.14.5 SDK (the version whose C++ vtable ABI `shader-slang-sys
		// 0.1.0` is hand-written against; the symbol was first added in Slang
		// v2025.6). This method is unused by both Ochroma and Spectra, so we
		// return None rather than link against a missing symbol. Restore the
		// original body if/when the SDK is upgraded to >= v2025.6.
		None
	}

	pub fn generic_container(&self) -> Option<&Generic> {
		rcall!(spReflectionVariable_GetGenericContainer(self) as Option<&Generic>)
	}

	pub fn apply_specializations(&self, generic: &Generic) -> Option<&Variable> {
		rcall!(
			spReflectionVariable_applySpecializations(self, generic as *const _ as *mut _)
				as Option<&Variable>
		)
	}
}
